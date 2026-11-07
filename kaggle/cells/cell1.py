%%writefile /kaggle/working/reid_common.py
"""CELL 1 (shared code): datasets (Market-1501, PRCC, LTCC), samplers, losses, evaluation, training loop,
checkpoint/resume. Paste everything from the %%writefile line down to the CELL 2 marker."""
import os, re, glob, json, math, time, random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = '/kaggle/working/outputs' if os.path.isdir('/kaggle/working') else './outputs'
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MAX_ITERS = None                      # Cell 2 sets this in QUICK mode to cut every epoch short
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================ dataset discovery + parsing
# every record is (image_path, person_id, camera_id[0-based], clothes_label[str])
MARKET_RE = re.compile(r'([-\d]+)_c(\d+)')            # 0002_c1s1_000451_03.jpg
LTCC_RE = re.compile(r'^(\d+)_(\d+)_c(\d+)')          # 001_1_c11_015366.png -> pid, outfit, camera
PRCC_CAMS = {'A': 0, 'B': 1, 'C': 2}                  # A, B: same clothes; C: different clothes
SKIP_DIRS = ('bounding_box', 'gt_', 'query', 'sketch', 'train', 'test', 'val', 'gallery')


def find_root(dataset, base='/kaggle/input'):
    """Locate an attached dataset anywhere under /kaggle/input. Returns None if it is not attached."""
    checks = {
        'market': lambda r, d: {'bounding_box_train', 'bounding_box_test', 'query'} <= d,
        'prcc': lambda r, d: os.path.basename(r) == 'rgb' and {'train', 'val', 'test'} <= d,
        'ltcc': lambda r, d: {'train', 'query', 'test'} <= d and any(
            LTCC_RE.match(n) for n in sorted(os.listdir(os.path.join(r, 'train')))[:20]),
    }
    for root, dirs, _ in os.walk(base):
        if checks[dataset](root, set(dirs)):
            return root
        dirs[:] = [d for d in dirs if not d.lower().startswith(SKIP_DIRS) and not d.isdigit()]
    return None


def parse_market(folder, keep_junk=False):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, '*.jpg'))):
        m = MARKET_RE.search(os.path.basename(p))
        if m is None:
            continue
        pid, cam = int(m.group(1)), int(m.group(2)) - 1
        if pid == -1 and not keep_junk:                  # -1 = junk image
            continue
        out.append((p, pid, cam, str(pid)))             # Market has no clothes labels: 1 outfit per id
    return out


def parse_ltcc(folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, '*.png'))):
        pid, clo, cam = map(int, LTCC_RE.match(os.path.basename(p)).groups())
        out.append((p, pid, cam - 1, f'{pid}_{clo}'))
    return out


def parse_prcc(folder, cam=None):
    """train/val: folder/<pid>/<A|B|C>_*.jpg   test: folder = test/<A|B|C>, cam given."""
    out = []
    for pdir in sorted(glob.glob(os.path.join(folder, '*'))):
        pid = int(os.path.basename(pdir))
        for p in sorted(glob.glob(os.path.join(pdir, '*.jpg'))):
            c = cam or os.path.basename(p)[0]
            out.append((p, pid, PRCC_CAMS[c], str(pid) if c in 'AB' else f'{pid}C'))
    return out


def load_dataset(name):
    """Raw records + the official test protocols of each dataset."""
    root = find_root(name)
    if root is None:
        raise FileNotFoundError(f'{name} is not attached. Add it with "+ Add Input" (see README).')
    j = lambda *a: os.path.join(root, *a)
    if name == 'market':
        train = parse_market(j('bounding_box_train'))
        sets = {'query': parse_market(j('query')), 'gallery': parse_market(j('bounding_box_test'))}
        tests, main = {'standard': ('query', 'gallery', 'general')}, 'standard'
    elif name == 'ltcc':
        train = parse_ltcc(j('train'))
        sets = {'query': parse_ltcc(j('query')), 'gallery': parse_ltcc(j('test'))}
        tests = {'general': ('query', 'gallery', 'general'), 'clothes_changing': ('query', 'gallery', 'cc')}
        main = 'clothes_changing'
    elif name == 'prcc':
        train = parse_prcc(j('train'))
        sets = {'gallery': parse_prcc(j('test', 'A'), 'A'), 'query_same': parse_prcc(j('test', 'B'), 'B'),
                'query_diff': parse_prcc(j('test', 'C'), 'C')}
        tests = {'same_clothes': ('query_same', 'gallery', 'general'),
                 'clothes_changing': ('query_diff', 'gallery', 'general')}
        main = 'clothes_changing'
    else:
        raise ValueError(name)
    return SimpleNamespace(name=name, root=root, train=train, sets=sets, tests=tests, main=main)


class ReIDDataset(Dataset):
    def __init__(self, data, transform, clothes=None):
        self.data, self.transform, self.clothes = data, transform, clothes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, pid, cam = self.data[i]
        return self.transform(Image.open(path).convert('RGB')), pid, cam, i


class RandomCroping:
    """CAL augmentation: with prob p, enlarge the image by 1/8 and randomly crop back to the original size."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        w, h = img.size
        nw, nh = int(round(w * 1.125)), int(round(h * 1.125))
        img = img.resize((nw, nh), Image.BILINEAR)
        x, y = int(round(random.uniform(0, nw - w))), int(round(random.uniform(0, nh - h)))
        return img.crop((x, y, x + w, y + h))


def build_transforms(size, train, mean=IMAGENET_MEAN, std=IMAGENET_STD, crop='pad', erasing=0.5, interp='bilinear'):
    mode = T.InterpolationMode.BICUBIC if interp == 'bicubic' else T.InterpolationMode.BILINEAR
    t = [T.Resize(size, interpolation=mode)]
    if train:
        if crop == 'cal':
            t.append(RandomCroping(0.5))
        t.append(T.RandomHorizontalFlip())
        if crop == 'pad':
            t += [T.Pad(10), T.RandomCrop(size)]
    t += [T.ToTensor(), T.Normalize(mean, std)]
    if train and erasing > 0:
        t.append(T.RandomErasing(p=erasing, scale=(0.02, 0.4), value=0))   # 0 = mean pixel after Normalize
    return T.Compose(t)


class RandomIdentitySampler(Sampler):
    """P identities x K instances per batch."""
    def __init__(self, data, batch_size, num_instances=4):
        self.K, self.P = num_instances, batch_size // num_instances
        self.index_dic = defaultdict(list)
        for i, (_, pid, _) in enumerate(data):
            self.index_dic[pid].append(i)
        self.pids = list(self.index_dic)
        self.length = sum(max(len(v), self.K) // self.K * self.K for v in self.index_dic.values())

    def __iter__(self):
        batches = defaultdict(list)
        for pid in self.pids:
            idxs = list(self.index_dic[pid])
            if len(idxs) < self.K:
                idxs = np.random.choice(idxs, self.K, replace=True).tolist()
            random.shuffle(idxs)
            for s in range(0, len(idxs) - self.K + 1, self.K):
                batches[pid].append(idxs[s:s + self.K])
        avail, out = list(self.pids), []
        while len(avail) >= self.P:
            for pid in random.sample(avail, self.P):
                out.extend(batches[pid].pop(0))
                if not batches[pid]:
                    avail.remove(pid)
        return iter(out)

    def __len__(self):
        return self.length


def build_data(dataset, size=(256, 128), batch_size=64, sampler='pk', num_instances=4, workers=4,
               mean=IMAGENET_MEAN, std=IMAGENET_STD, crop='pad', erasing=0.5, interp='bilinear', test_batch=128):
    ds = load_dataset(dataset)
    pid_lut = {p: i for i, p in enumerate(sorted({r[1] for r in ds.train}))}
    clo_lut = {c: i for i, c in enumerate(sorted({r[3] for r in ds.train}))}
    train = [(p, pid_lut[pid], cam) for p, pid, cam, _ in ds.train]
    train_clothes = torch.tensor([clo_lut[r[3]] for r in ds.train])
    pid2clothes = torch.zeros(len(pid_lut), len(clo_lut))           # pid2clothes[i, j] = 1 if outfit j belongs to id i
    pid2clothes[torch.tensor([x[1] for x in train]), train_clothes] = 1
    test_clo = {c: i for i, c in enumerate(sorted({r[3] for s in ds.sets.values() for r in s}))}

    tr_tf = build_transforms(size, True, mean, std, crop, erasing, interp)
    te_tf = build_transforms(size, False, mean, std, interp=interp)
    kw = dict(num_workers=workers, pin_memory=True)
    if sampler == 'pk':
        train_loader = DataLoader(ReIDDataset(train, tr_tf), batch_size, drop_last=True,
                                  sampler=RandomIdentitySampler(train, batch_size, num_instances), **kw)
    else:
        train_loader = DataLoader(ReIDDataset(train, tr_tf), batch_size, shuffle=True, drop_last=True, **kw)
    loaders = {k: DataLoader(ReIDDataset([(p, pid, cam) for p, pid, cam, _ in v], te_tf,
                                         clothes=np.array([test_clo[r[3]] for r in v])), test_batch, shuffle=False, **kw)
               for k, v in ds.sets.items()}
    all_recs = ds.train + [r for s in ds.sets.values() for r in s]
    d = SimpleNamespace(name=dataset, train=train, train_clothes=train_clothes, pid2clothes=pid2clothes,
                        train_loader=train_loader, main_test=ds.main,
                        tests={k: (loaders[q], loaders[g], m) for k, (q, g, m) in ds.tests.items()},
                        num_classes=len(pid_lut), num_clothes=len(clo_lut), num_cams=1 + max(r[2] for r in all_recs))
    print(f'[{dataset}] train {len(train)} imgs / {d.num_classes} ids / {d.num_clothes} outfits | ' +
          ' | '.join(f'{k} {len(v)} imgs' for k, v in ds.sets.items()) + f' | cameras {d.num_cams}')
    return d


# ============================================================ losses
def euclidean_dist(x, y):
    xx, yy = x.pow(2).sum(1, keepdim=True), y.pow(2).sum(1, keepdim=True).t()
    return (xx + yy - 2 * x @ y.t()).clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """Batch-hard triplet loss. margin=None -> soft-margin version (used by TransReID)."""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels):
        d = euclidean_dist(feats.float(), feats.float())
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        d_ap = d.masked_fill(~same, -1).max(1)[0]              # hardest positive
        d_an = d.masked_fill(same, float('inf')).min(1)[0]     # hardest negative
        y = torch.ones_like(d_an)
        if self.margin is None:
            return F.soft_margin_loss(d_an - d_ap, y)
        return F.margin_ranking_loss(d_an, d_ap, y, margin=self.margin)


# ============================================================ evaluation
@torch.no_grad()
def extract_features(model, loader, flip=False, forward_fn=None):
    forward_fn = forward_fn or (lambda m, x, c: m(x))
    model.eval()
    feats, pids, cams = [], [], []
    for imgs, pid, cam, _ in loader:
        imgs, cam_d = imgs.to(DEVICE, non_blocking=True), cam.to(DEVICE)
        with autocast():
            f = forward_fn(model, imgs, cam_d).float()
            if flip:
                f = f + forward_fn(model, torch.flip(imgs, [3]), cam_d).float()
        feats.append(f.cpu()); pids.append(pid); cams.append(cam)
    return torch.cat(feats), torch.cat(pids).numpy(), torch.cat(cams).numpy()


@torch.no_grad()
def cosine_distmat(qf, gf):
    qf, gf = F.normalize(qf, dim=1).to(DEVICE), F.normalize(gf, dim=1).to(DEVICE)
    return torch.cat([(2 - 2 * qf[i:i + 1024] @ gf.t()).cpu() for i in range(0, len(qf), 1024)]).numpy()


def evaluate_rank(distmat, q_pids, g_pids, q_cams, g_cams, q_clo=None, g_clo=None, mode='general', max_rank=50):
    """general: ignore gallery images of the same id AND same camera (standard protocol).
    cc: additionally ignore same-id images wearing the same clothes (clothes-changing protocol).
    sc: additionally ignore same-id images wearing different clothes (same-clothes protocol)."""
    indices = np.argsort(distmat, axis=1)
    all_cmc, all_ap, all_inp = [], [], []
    for q in range(len(q_pids)):
        order = indices[q]
        same_id = g_pids[order] == q_pids[q]
        junk = same_id & (g_cams[order] == q_cams[q])
        if mode == 'cc':
            junk |= same_id & (g_clo[order] == q_clo[q])
        elif mode == 'sc':
            junk |= same_id & (g_clo[order] != q_clo[q])
        raw = same_id[~junk].astype(np.float32)
        if not raw.any():
            continue
        cum = raw.cumsum()
        all_inp.append(raw.sum() / (np.where(raw == 1)[0].max() + 1.0))
        all_cmc.append(np.minimum(cum, 1)[:max_rank])
        all_ap.append(((cum / (np.arange(len(raw)) + 1.0)) * raw).sum() / raw.sum())
    return np.mean(all_cmc, 0), float(np.mean(all_ap)), float(np.mean(all_inp))


def test(model, data, flip=False, forward_fn=None):
    """Runs every protocol of the dataset. Top-level keys (mAP, R1, ...) = main protocol (used to pick 'best')."""
    t0, cache, res = time.time(), {}, {}

    def feats(loader):
        if id(loader) not in cache:
            cache[id(loader)] = extract_features(model, loader, flip, forward_fn)
        return cache[id(loader)]

    for name, (ql, gl, mode) in data.tests.items():
        (qf, qp, qc), (gf, gp, gc) = feats(ql), feats(gl)
        cmc, mAP, mINP = evaluate_rank(cosine_distmat(qf, gf), qp, gp, qc, gc, ql.dataset.clothes, gl.dataset.clothes, mode)
        r = {'mAP': mAP, 'R1': float(cmc[0]), 'R5': float(cmc[4]), 'R10': float(cmc[9]), 'mINP': mINP}
        print(f'   TEST [{name:16s}] ' + '  '.join(f'{k}={v * 100:.2f}' for k, v in r.items()))
        res.update({f'{name}/{k}': v for k, v in r.items()})
        if name == data.main_test:
            res.update(r)
    print(f'   (test time {time.time() - t0:.0f}s)')
    return res


# ============================================================ training helpers
def autocast():
    return torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda')


def make_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')


def warmup_cosine(optimizer, total_epochs, warmup_epochs=5, warmup_factor=0.01, min_ratio=0.002):
    def f(e):
        if e < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * e / warmup_epochs
        p = (e - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# ============================================================ checkpoint / resume / time budget
START_TIME, MAX_HOURS, STOPPED = time.time(), 11.5, False      # Kaggle kills sessions at 12 h


def time_left():
    return MAX_HOURS * 3600 - (time.time() - START_TIME)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_ckpt(name, **state):
    """Atomic save (write tmp, then rename) so a crash during saving never corrupts the checkpoint."""
    path = f'{OUT_DIR}/{name}_ckpt.pth'
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def load_ckpt(name):
    path = f'{OUT_DIR}/{name}_ckpt.pth'
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location='cpu', weights_only=False)
    print(f'>> RESUMING {name} from epoch {ck["epoch"]}')
    return ck


def restore_previous_outputs(base='/kaggle/input'):
    """Copy checkpoints/results of an attached earlier notebook version (its 'outputs' folder) into OUT_DIR."""
    import shutil
    n = 0
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d == 'outputs' or (not d.lower().startswith(SKIP_DIRS) and not d.isdigit())]
        if os.path.basename(root) != 'outputs' or not any(f.endswith(('_ckpt.pth', '_history.json')) for f in files):
            continue
        for f in files:
            dst = os.path.join(OUT_DIR, f)
            if not f.startswith('quick_') and not f.endswith('.tmp') and not os.path.exists(dst):
                shutil.copy2(os.path.join(root, f), dst); n += 1
                print('   restored', f, 'from', root)
    print(f'restored {n} file(s) from previous runs' if n else 'no previous outputs attached (fresh start)')


def fit(model, data, optimizer, scheduler, step_fn, epochs, name, eval_every=10,
        flip=False, test_forward=None, start_hook=None, select_last=False):
    """Generic epoch loop. step_fn(model, imgs, pids, cams, idx) -> loss.
    Saves a full checkpoint every epoch, resumes from it automatically, and stops cleanly before the time limit.
    select_last=True: the reported model is the LAST evaluated one (no model selection on the test set)."""
    global STOPPED
    model.to(DEVICE)
    scaler = make_scaler()
    history, best, start, eval_time = [], {'mAP': -1}, 0, 0.0
    ck = load_ckpt(name)
    if ck:
        unwrap(model).load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        history, best, start = ck['history'], ck['best'], ck['epoch']
        if start >= epochs:
            print(f'[{name}] already finished, skipping training')
            return history, best
    for ep in range(start, epochs):
        if start_hook:
            start_hook(model, ep)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for imgs, pids, cams, idx in data.train_loader:
            imgs, pids, cams = imgs.to(DEVICE, non_blocking=True), pids.to(DEVICE), cams.to(DEVICE)
            with autocast():
                loss = step_fn(model, imgs, pids, cams, idx)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tot += loss.item(); n += 1
            if MAX_ITERS and n >= MAX_ITERS:
                break
        rec = {'epoch': ep + 1, 'loss': tot / max(n, 1), 'lr': optimizer.param_groups[0]['lr'], 'time': time.time() - t0}
        scheduler.step()
        print(f"[{name}] epoch {ep + 1}/{epochs}  loss {rec['loss']:.4f}  lr {rec['lr']:.2e}  {rec['time']:.0f}s")
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            t1 = time.time()
            rec.update(test(model, data, flip, test_forward))
            eval_time = time.time() - t1
            if select_last or rec['mAP'] > best['mAP']:
                best = dict(rec)
                torch.save(unwrap(model).state_dict(), f'{OUT_DIR}/{name}_best.pth')
        history.append(rec)
        save_ckpt(name, model=unwrap(model).state_dict(), optimizer=optimizer.state_dict(),
                  scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1, history=history, best=best)
        json.dump({'history': history, 'best': best}, open(f'{OUT_DIR}/{name}_history.json', 'w'), indent=1)
        if ep + 1 < epochs and time_left() < 1.5 * rec['time'] + eval_time + 300:
            STOPPED = True
            print(f'[{name}] TIME BUDGET REACHED after epoch {ep + 1}: checkpoint saved, stopping cleanly.')
            break
    return history, best


# ============================================================ reporting (plain black/grey figures)
def plot_history(history, name):
    import matplotlib.pyplot as plt
    with plt.style.context('grayscale'):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].plot([h['epoch'] for h in history], [h['loss'] for h in history], color='black')
        ax[0].set(title=f'{name}: training loss', xlabel='epoch')
        ev = [h for h in history if 'mAP' in h]
        for k, style in (('mAP', 'o-'), ('R1', 's--')):
            ax[1].plot([h['epoch'] for h in ev], [h[k] * 100 for h in ev], style, color='black', label=k)
        ax[1].set(title=f'{name}: test accuracy (main protocol)', xlabel='epoch', ylabel='%'); ax[1].legend()
        plt.tight_layout(); plt.savefig(f'{OUT_DIR}/{name}_curves.png', dpi=150); plt.show()


def compare(method, best, data, paper=None):
    """Prints our best result for every protocol next to the paper's numbers."""
    paper = paper or {}
    print(f"\n{method} on {data.name} (best epoch {best.get('epoch')})")
    print(f"{'protocol':18s}{'R1':>8s}{'mAP':>8s}{'mINP':>8s}   |  paper R1 / mAP")
    for name in data.tests:
        if f'{name}/R1' in best:
            p = paper.get(name, ('n/a', 'n/a'))
            print(f"{name:18s}{best[f'{name}/R1'] * 100:8.2f}{best[f'{name}/mAP'] * 100:8.2f}"
                  f"{best[f'{name}/mINP'] * 100:8.2f}   |  {p[0]} / {p[1]}")
