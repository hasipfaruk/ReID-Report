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


# ===== CELL 2 : SWITCH (the only cell you edit) =====
JOB = 'transreid_prcc'
#   'explore'           dataset statistics + sample images of every attached dataset (2 min)
#   'transreid_market'  PAPER 1  TransReID (ICCV 2021) reproduction on Market-1501
#   'cal_prcc'          PAPER 2  CAL (CVPR 2022) reproduction on PRCC
#   'baseline_prcc'     CAL ablation: same network trained WITHOUT the clothes-adversarial loss
#   'transreid_prcc'    TransReID on clothes-changing PRCC (comparison between the two papers)
#   'cal_ltcc', 'baseline_ltcc', 'transreid_ltcc'   only if you obtained LTCC
#   'all'               with QUICK=True: smoke test of every job whose dataset is attached
#   ---- FAIR COMPARISON on PRCC clothes-changing (supervisor's request): same data, input size, batch,
#        augmentation, epochs and test protocol for all three models; last epoch is reported.
#   'fair_baseline'     ResNet-50 baseline (no CAL)            ~1.5 h
#   'fair_cal'          ResNet-50 + CAL                        ~1.5 h
#   'fair_transreid'    TransReID (ViT-B/16 + SIE + JPM)       ~3.5 h
#   'fair_all'          the three above, one after the other   ~6.5 h  (fits in one 12 h session)
QUICK = False      # True = smoke test (30 iterations per epoch, 2 epochs); False = real training run
MAX_HOURS = 11.5   # training stops cleanly (checkpoint saved) before this; Kaggle's hard limit is 12 h
SEED = 0           # fair jobs: run with SEED = 0, then 1 and 2 if GPU quota allows (mean +- std in Cell 7)

import sys, os, re, gc, copy, math, json, time, glob, random, subprocess, importlib, collections, traceback
sys.path.insert(0, '/kaggle/working')
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image
import reid_common as rc; importlib.reload(rc)

rc.set_seed(SEED)
rc.MAX_HOURS = MAX_HOURS
# FAIR protocol shared by the three fair_* jobs (everything that is NOT part of a method is identical)
FAIR = dict(size=(256, 128), batch_size=64, num_instances=4, crop='pad', erasing=0.5, interp='bilinear',
            epochs=60, eval_every=10, flip_test=True)
rc.MAX_ITERS = 30 if QUICK else None
tag = lambda name: ('quick_' if QUICK else '') + name          # quick-test outputs never overwrite real ones
if QUICK:                                                      # smoke tests always start from scratch
    for f in glob.glob(f'{rc.OUT_DIR}/quick_*'):
        os.remove(f)
else:                                                          # pick up checkpoints of an attached earlier version
    rc.restore_previous_outputs()
ATTACHED = {d: rc.find_root(d) for d in ('market', 'prcc', 'ltcc')}
for d, r in ATTACHED.items():
    print(f'{d:7s}: {r or "NOT attached"}')
print(f'JOB={JOB}  QUICK={QUICK}  torch {torch.__version__}  GPUs: {torch.cuda.device_count()}',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')


# ===== CELL 3 : DATASET EXPLORATION =====
def show_images(items, title):
    """items: list of (path, caption)."""
    if not items:
        return
    fig, ax = plt.subplots(1, len(items), figsize=(1.5 * len(items), 3.4))
    for a, (p, cap) in zip(np.atleast_1d(ax), items):
        a.imshow(Image.open(p)); a.set_title(cap, fontsize=7); a.axis('off')
    fig.suptitle(title, fontsize=10); plt.show()


def run_explore():
    stats = {}
    for name in ('market', 'prcc', 'ltcc'):
        if ATTACHED[name] is None:
            print(f'\n{name}: not attached, skipped'); continue
        ds = rc.load_dataset(name)
        splits = {'train': ds.train, **ds.sets}
        print(f'\n=== {name.upper()}  ({ds.root})')
        print(f"{'split':12s}{'images':>8s}{'IDs':>6s}{'cams':>6s}{'outfits':>9s}{'img/ID':>8s}{'outfits/ID':>12s}")
        stats[name] = {}
        for s, recs in splits.items():
            recs = [r for r in recs if r[1] > 0] if name == 'market' else recs   # drop Market distractors/junk
            by_id = collections.defaultdict(set)
            for r in recs:
                by_id[r[1]].add(r[3])
            st = stats[name][s] = dict(images=len(recs), ids=len(by_id), cams=len({r[2] for r in recs}),
                                       outfits=len({r[3] for r in recs}), img_per_id=round(len(recs) / len(by_id), 1),
                                       outfits_per_id=round(np.mean([len(v) for v in by_id.values()]), 2))
            print(f"{s:12s}{st['images']:8d}{st['ids']:6d}{st['cams']:6d}{st['outfits']:9d}{st['img_per_id']:8.1f}{st['outfits_per_id']:12.2f}")
        if name == 'market':
            g = [r[1] for r in ds.sets['gallery']]
            print(f"gallery distractors (id 0): {g.count(0)}   (junk id -1 images are removed)")
        test_ids = {r[1] for s in ds.sets.values() for r in s}
        print('train/test identity overlap:', len({r[1] for r in ds.train} & test_ids), '(must be 0)')

        # distributions (black/grey only)
        cnt = collections.Counter(r[1] for r in ds.train)
        outfits = collections.defaultdict(set)
        for r in ds.train:
            outfits[r[1]].add(r[3])
        with plt.style.context('grayscale'):
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
            ax[0].hist(list(cnt.values()), bins=30, color='0.4', edgecolor='black')
            ax[0].set(title=f'{name} train: images per identity', xlabel='#images', ylabel='#identities')
            cams = collections.Counter(r[2] + 1 for r in ds.train)
            ax[1].bar(sorted(cams), [cams[c] for c in sorted(cams)], color='0.4', edgecolor='black')
            ax[1].set(title=f'{name} train: images per camera', xlabel='camera')
            n_out = [len(v) for v in outfits.values()]
            ax[2].hist(n_out, bins=np.arange(1, max(n_out) + 2) - 0.5, rwidth=0.8, color='0.4', edgecolor='black')
            ax[2].set(title=f'{name} train: outfits per identity', xlabel='#outfits')
            plt.tight_layout(); plt.savefig(f'{rc.OUT_DIR}/{name}_stats.png', dpi=150); plt.show()

        # one identity: for clothes-changing datasets group by outfit, for Market by camera
        random.seed(1)
        by_id = collections.defaultdict(list)
        for r in ds.train:
            by_id[r[1]].append(r)
        if name == 'market':
            pid = max(by_id, key=lambda p: len({r[2] for r in by_id[p]}))      # id seen by the most cameras
            items = [(r[0], f'cam {r[2] + 1}') for r in sorted({r[2]: r for r in by_id[pid]}.values(), key=lambda r: r[2])]
            show_images(items, f'Market-1501: identity {pid} seen by different cameras')
        else:
            pid = max(by_id, key=lambda p: len({r[3] for r in by_id[p]}))
            groups = collections.defaultdict(list)
            for r in by_id[pid]:
                groups[r[3]].append(r)
            items = [(r[0], f'outfit {k + 1}\ncam {r[2] + 1}') for k, g in enumerate(list(groups.values())[:4])
                     for r in random.sample(g, min(3, len(g)))]
            show_images(items, f'{name.upper()}: identity {pid}, same person in different clothes')
    json.dump(stats, open(f'{rc.OUT_DIR}/dataset_stats.json', 'w'), indent=1)


# ===== CELL 4 : PAPER 1  TransReID (He et al., ICCV 2021) =====
# ViT-B/16 + overlapping patches (stride 12) + SIE (camera embedding) + JPM (jigsaw patch module).
# Settings follow the official configs/Market/vit_transreid_stride.yml.
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm'], check=False)
    import timm


def shuffle_unit(x, shift, groups):
    """JPM: shift the patch sequence (drop cls) then patch-shuffle it into `groups` interleaved groups."""
    B, _, D = x.shape
    x = torch.cat([x[:, shift:], x[:, 1:shift]], dim=1)
    if x.size(1) % groups:
        x = torch.cat([x, x[:, -2:-1].expand(B, groups - x.size(1) % groups, D)], dim=1)
    return x.view(B, groups, -1, D).transpose(1, 2).reshape(B, -1, D)


class TransReID(nn.Module):
    def __init__(self, num_classes, num_cams, img_size=(256, 128), stride=12, sie_coe=3.0,
                 jpm=True, shift=5, groups=2, divide=4, drop_path=0.1):
        super().__init__()
        try:   # jx_vit_base_p16_224 (ImageNet-21k pre-trained, 1k fine-tuned) = weights used by the paper
            base = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, drop_path_rate=drop_path)
        except Exception as e:
            print('falling back to default vit_base_patch16_224 weights:', e)
            base = timm.create_model('vit_base_patch16_224', pretrained=True, drop_path_rate=drop_path)
        D = base.embed_dim
        self.grid = ((img_size[0] - 16) // stride + 1, (img_size[1] - 16) // stride + 1)   # 21 x 11 patches
        self.patch_proj = nn.Conv2d(3, D, 16, stride)                                       # overlapping patches
        self.patch_proj.load_state_dict(base.patch_embed.proj.state_dict())
        self.cls_token = base.cls_token
        pe = base.pos_embed.data                                                            # 1 x (1+14*14) x D
        g0 = int(math.sqrt(pe.shape[1] - 1))
        grid_pe = F.interpolate(pe[:, 1:].reshape(1, g0, g0, D).permute(0, 3, 1, 2), size=self.grid,
                                mode='bilinear', align_corners=False).permute(0, 2, 3, 1).reshape(1, -1, D)
        self.pos_embed = nn.Parameter(torch.cat([pe[:, :1], grid_pe], 1))
        self.sie_coe = sie_coe
        self.sie_embed = nn.Parameter(torch.zeros(num_cams, 1, D))
        nn.init.trunc_normal_(self.sie_embed, std=0.02)
        self.blocks = base.blocks[:-1]                                                      # first 11 blocks
        self.b1 = nn.Sequential(copy.deepcopy(base.blocks[-1]), copy.deepcopy(base.norm))   # global branch
        self.jpm, self.shift, self.groups, self.divide = jpm, shift, groups, divide
        if jpm:
            self.b2 = nn.Sequential(copy.deepcopy(base.blocks[-1]), copy.deepcopy(base.norm))  # shared local branch
        n = 1 + (divide if jpm else 0)
        self.bnnecks = nn.ModuleList([nn.BatchNorm1d(D) for _ in range(n)])
        self.classifiers = nn.ModuleList([nn.Linear(D, num_classes, bias=False) for _ in range(n)])
        for bn in self.bnnecks:
            nn.init.ones_(bn.weight); nn.init.zeros_(bn.bias); bn.bias.requires_grad_(False)
        for c in self.classifiers:
            nn.init.normal_(c.weight, std=0.001)

    def forward(self, x, cams):
        B = x.size(0)
        x = self.patch_proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1)
        x = x + self.pos_embed + self.sie_coe * self.sie_embed[cams]      # SIE: add camera embedding
        x = self.blocks(x)
        feats = [self.b1(x)[:, 0]]
        if self.jpm:
            L = (x.size(1) - 1) // self.divide
            tok, xs = x[:, :1], shuffle_unit(x, self.shift, self.groups)
            feats += [self.b2(torch.cat([tok, xs[:, i * L:(i + 1) * L]], 1))[:, 0] for i in range(self.divide)]
        if not self.training:      # NECK_FEAT 'before': global + local/4 features before BN
            return torch.cat([feats[0]] + [f / self.divide for f in feats[1:]], 1)
        logits = [c(bn(f)) for f, bn, c in zip(feats, self.bnnecks, self.classifiers)]
        return logits, feats


triplet_soft = rc.TripletLoss(margin=None)     # NO_MARGIN: True in the official config


def transreid_step(model, imgs, pids, cams, idx):
    logits, feats = model(imgs, cams)
    ce = [F.cross_entropy(l.float(), pids) for l in logits]          # IF_LABELSMOOTH: off
    tri = [triplet_soft(f, pids) for f in feats]
    if len(ce) == 1:
        return ce[0] + tri[0]
    # global branch weight 0.5, mean of the JPM local branches weight 0.5
    return 0.5 * (ce[0] + tri[0]) + 0.5 * (sum(ce[1:]) + sum(tri[1:])) / (len(ce) - 1)


TRANSREID_PAPER = {'market': {'standard': (95.2, 88.9)}}     # TransReID* (stride 12), no re-ranking


def run_transreid(dataset, name, fair=False):
    # 256x128, flip + pad 10/crop + random erasing 0.5, 16 ids x 4 imgs, mean/std 0.5,
    # SGD lr 0.008 (bias x2), wd 1e-4, 5 warm-up epochs + cosine, 120 epochs, no flip at test
    # fair=True: FAIR protocol (60 epochs, flip at test, last epoch reported); optimiser and losses unchanged.
    # mean/std 0.5 is kept: it is the normalisation the ViT weights were pre-trained with.
    if fair:
        data = rc.build_data(dataset, size=FAIR['size'], batch_size=FAIR['batch_size'], sampler='pk',
                             num_instances=FAIR['num_instances'], mean=[0.5] * 3, std=[0.5] * 3,
                             crop=FAIR['crop'], erasing=FAIR['erasing'], interp=FAIR['interp'])
    else:
        data = rc.build_data(dataset, size=(256, 128), batch_size=64, sampler='pk', num_instances=4,
                             mean=[0.5] * 3, std=[0.5] * 3, crop='pad', erasing=0.5, interp='bicubic')
    EPOCHS, BASE_LR = (2 if QUICK else (FAIR['epochs'] if fair else 120)), 0.008
    model = TransReID(data.num_classes, data.num_cams)
    groups = [{'params': [p], 'lr': BASE_LR * (2 if 'bias' in n else 1), 'weight_decay': 1e-4}
              for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD(groups, lr=BASE_LR, momentum=0.9)
    sched = rc.warmup_cosine(opt, EPOCHS, warmup_epochs=5)
    model = model.to(rc.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)                     # uses both T4 GPUs
    hist, best = rc.fit(model, data, opt, sched, transreid_step, EPOCHS, name,
                        eval_every=FAIR['eval_every'] if fair else 20, flip=fair and FAIR['flip_test'],
                        test_forward=lambda m, x, c: m(x, c), select_last=fair)
    if rc.STOPPED:
        return
    rc.plot_history(hist, name)
    rc.compare('TransReID' + (' (fair protocol)' if fair else ''), best, data,
               None if fair else TRANSREID_PAPER.get(dataset))


# ===== CELL 5 : PAPER 2  CAL, Clothes-based Adversarial Loss (Gu et al., CVPR 2022) =====
# Settings follow the official Simple-CCReID code (configs/default_img.py + res50_cels_cal.yaml).
class CALNet(nn.Module):
    """ResNet-50 (last stride 1) -> concat(global max pool, global avg pool) = 4096-d -> BN.
    classifier: identity classifier (linear); clothes_classifier: cosine classifier over all training outfits."""
    def __init__(self, num_pids, num_clothes, dim=4096):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(r.children())[:-2])
        self.bn = nn.BatchNorm1d(dim)
        nn.init.normal_(self.bn.weight, 1.0, 0.02); nn.init.zeros_(self.bn.bias)
        self.classifier = nn.Linear(dim, num_pids)
        nn.init.normal_(self.classifier.weight, std=0.001); nn.init.zeros_(self.classifier.bias)
        self.clothes_weight = nn.Parameter(torch.empty(num_clothes, dim).uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5))

    def forward(self, x):
        x = self.base(x)
        return self.bn(torch.cat([F.adaptive_max_pool2d(x, 1), F.adaptive_avg_pool2d(x, 1)], 1).flatten(1))

    def clothes_cos(self, f, detach_weight=False):
        w = self.clothes_weight.detach() if detach_weight else self.clothes_weight
        return F.normalize(f, dim=1) @ F.normalize(w, dim=1).t()


def clothes_adversarial_loss(cos, targets, pos_mask, scale=16.0, epsilon=0.1):
    """Eq. of the CAL paper: pull the feature towards ALL outfits of its own identity (positives), push away from
    outfits of other identities, so the backbone cannot tell outfits of the same person apart."""
    x = scale * cos
    neg_mask = 1 - pos_mask
    id_mask = F.one_hot(targets, x.size(1)).float()
    exp = torch.exp(x)
    log_prob = x - torch.log((exp * neg_mask).sum(1, keepdim=True) + exp)
    mask = (1 - epsilon) * id_mask + epsilon / pos_mask.sum(1, keepdim=True) * pos_mask
    return (-mask * log_prob).sum(1).mean()


CAL_STATE = {}


def cal_step(model, imgs, pids, cams, idx):
    st = CAL_STATE
    f = model(imgs)
    loss = F.cross_entropy(model.classifier(f).float(), pids, label_smoothing=0.1)    # identity loss (CE + LS)
    if st['adversarial'] and st['epoch'] >= st['start_cc']:
        clothes = st['clothes'][idx].to(rc.DEVICE)
        with torch.autocast('cuda', enabled=False):
            f32 = f.float()
            # (1) clothes classifier: CosFace(s=16, m=0) on DETACHED features -> only the classifier learns
            loss = loss + F.cross_entropy(16.0 * model.clothes_cos(f32.detach()), clothes)
            # (2) adversarial loss on the backbone, classifier weights frozen for this term
            if st['epoch'] >= st['start_adv']:
                loss = loss + clothes_adversarial_loss(model.clothes_cos(f32, detach_weight=True),
                                                       clothes, st['pid2clothes'][pids])
    return loss


CAL_PAPER = {'prcc': {'same_clothes': (100.0, 99.8), 'clothes_changing': (55.2, 55.8)},
             'ltcc': {'general': (74.2, 40.8), 'clothes_changing': (40.1, 18.0)}}


def run_cal(dataset, adversarial, name, fair=False):
    # 384x192, random crop 0.5 + flip 0.5 + random erasing 0.5, batch 32 (8 imgs per id),
    # Adam lr 3.5e-4 wd 5e-4, lr x0.1 at epochs 20 and 40 (clothes classifier keeps a constant lr),
    # 60 epochs, clothes classifier + adversarial loss from epoch 25, flip at test, eval every 5 epochs
    # fair=True: FAIR protocol (input size, batch, augmentation, last epoch reported); optimiser and losses unchanged.
    if fair:
        data = rc.build_data(dataset, size=FAIR['size'], batch_size=FAIR['batch_size'], sampler='pk',
                             num_instances=FAIR['num_instances'], crop=FAIR['crop'], erasing=FAIR['erasing'],
                             interp=FAIR['interp'])
    else:
        data = rc.build_data(dataset, size=(384, 192), batch_size=32, sampler='pk', num_instances=8,
                             crop='cal', erasing=0.5)
    EPOCHS, START = (2, 1) if QUICK else ((FAIR['epochs'], 25) if fair else (60, 25))
    model = CALNet(data.num_classes, data.num_clothes)
    CAL_STATE.update(adversarial=adversarial, start_cc=START, start_adv=START, epoch=0,
                     clothes=data.train_clothes, pid2clothes=data.pid2clothes.to(rc.DEVICE))
    backbone = [p for n, p in model.named_parameters() if n != 'clothes_weight']
    opt = torch.optim.Adam([{'params': backbone}, {'params': [model.clothes_weight]}], lr=3.5e-4, weight_decay=5e-4)
    step_decay = lambda e: 0.1 ** sum(e >= m for m in (20, 40))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [step_decay, lambda e: 1.0])
    hist, best = rc.fit(model, data, opt, sched, cal_step, EPOCHS, name,
                        eval_every=FAIR['eval_every'] if fair else 5, flip=True,
                        start_hook=lambda m, ep: CAL_STATE.update(epoch=ep), select_last=fair)
    if rc.STOPPED:
        return
    rc.plot_history(hist, name)
    rc.compare(('CAL' if adversarial else 'Baseline (no CAL)') + (' (fair protocol)' if fair else ''), best, data,
               CAL_PAPER.get(dataset) if adversarial and not fair else None)


# ===== CELL 6 : RUN =====
JOBS = {
    'explore': (None, run_explore),
    'transreid_market': ('market', lambda n: run_transreid('market', n)),
    'cal_prcc': ('prcc', lambda n: run_cal('prcc', True, n)),
    'baseline_prcc': ('prcc', lambda n: run_cal('prcc', False, n)),
    'transreid_prcc': ('prcc', lambda n: run_transreid('prcc', n)),
    'cal_ltcc': ('ltcc', lambda n: run_cal('ltcc', True, n)),
    'baseline_ltcc': ('ltcc', lambda n: run_cal('ltcc', False, n)),
    'transreid_ltcc': ('ltcc', lambda n: run_transreid('ltcc', n)),
    # fair comparison on PRCC (one name per seed, e.g. fair_cal_s0)
    f'fair_baseline_s{SEED}': ('prcc', lambda n: run_cal('prcc', False, n, fair=True)),
    f'fair_cal_s{SEED}': ('prcc', lambda n: run_cal('prcc', True, n, fair=True)),
    f'fair_transreid_s{SEED}': ('prcc', lambda n: run_transreid('prcc', n, fair=True)),
}
FAIR_JOBS = [f'fair_baseline_s{SEED}', f'fair_cal_s{SEED}', f'fair_transreid_s{SEED}']
if JOB == 'all':
    todo = list(JOBS)
elif JOB == 'fair_all':
    todo = FAIR_JOBS
elif JOB in ('fair_baseline', 'fair_cal', 'fair_transreid'):
    todo = [f'{JOB}_s{SEED}']
else:
    todo = [JOB]
if JOB == 'all' and not QUICK:
    print('WARNING: full training of all jobs exceeds the 12 h Kaggle limit. Run one job per commit.')
for job in todo:
    dataset, fn = JOBS[job]
    if dataset and ATTACHED[dataset] is None:
        print(f'\n--- {job}: skipped ({dataset} not attached)'); continue
    print(f'\n{"=" * 30} {job.upper()} {"=" * 30}')
    t0 = time.time()
    try:
        fn() if dataset is None else fn(tag(job))
    except Exception:
        # do not crash the notebook: a finished version keeps all checkpoints/results in its Output tab
        err = traceback.format_exc()
        print(err)
        open(f'{rc.OUT_DIR}/ERROR_{job}.txt', 'w').write(err)
        print(f'!! {job} FAILED. Error saved to outputs/ERROR_{job}.txt (paste it in the chat). Checkpoints are kept.')
        if not QUICK:
            break
    gc.collect(); torch.cuda.empty_cache()
    print(f'{job} done in {(time.time() - t0) / 60:.1f} min')
    if rc.STOPPED:
        print('\n>> STOPPED BEFORE THE TIME LIMIT. To continue: "+ Add Input" -> Your Work -> Notebooks -> '
              'this notebook (latest version), keep the same JOB, then Save & Run All again.')
        break
print('\nOutputs saved in', rc.OUT_DIR, ':', sorted(os.listdir(rc.OUT_DIR)))



# ===== CELL 7 : REPORT 1  results table (CSV) + comparison charts + sample person images =====
# Reads the *_history.json files in outputs/. Run it after the training jobs (see README, "Report run").
import csv

REPORT_JOBS = ['transreid_market', 'cal_prcc', 'baseline_prcc', 'transreid_prcc',
               'cal_ltcc', 'baseline_ltcc', 'transreid_ltcc']
METHOD_NAME = {'transreid': 'TransReID', 'cal': 'CAL', 'baseline': 'Baseline (no CAL)'}
PAPER_NUMBERS = {'transreid': TRANSREID_PAPER, 'cal': CAL_PAPER, 'baseline': {}}


def collect_results():
    rows = []
    for job in REPORT_JOBS:
        path = f'{rc.OUT_DIR}/{tag(job)}_history.json'
        if not os.path.exists(path):
            continue
        h = json.load(open(path))
        best = h['best']
        method, dataset = job.split('_')
        if 'R1' not in best:
            continue
        hours = sum(r['time'] for r in h['history']) / 3600
        for key in best:
            if key.endswith('/R1'):
                prot = key[:-3]
                p = PAPER_NUMBERS[method].get(dataset, {}).get(prot, ('', ''))
                rows.append(dict(method=METHOD_NAME[method], dataset=dataset, protocol=prot,
                                 **{m: round(best[f'{prot}/{m}'] * 100, 2) for m in ('R1', 'R5', 'R10', 'mAP', 'mINP')},
                                 paper_R1=p[0], paper_mAP=p[1], best_epoch=best['epoch'],
                                 epochs_run=len(h['history']), train_hours=round(hours, 2)))
    return rows


def results_charts(rows):
    """One grey/black grouped bar chart per dataset: methods x protocols, Rank-1 and mAP."""
    hatches = ['', '//', '..', 'xx']
    for dataset in sorted({r['dataset'] for r in rows}):
        rs = [r for r in rows if r['dataset'] == dataset]
        prots = list(dict.fromkeys(r['protocol'] for r in rs))
        methods = list(dict.fromkeys(r['method'] for r in rs))
        w = 0.8 / len(methods)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, metric in zip(axes, ('R1', 'mAP')):
            for i, m in enumerate(methods):
                vals = [next((r[metric] for r in rs if r['method'] == m and r['protocol'] == p), 0) for p in prots]
                bars = ax.bar(np.arange(len(prots)) + (i - (len(methods) - 1) / 2) * w, vals, w, label=m,
                              color=str(0.35 + 0.2 * (i % 3)), edgecolor='black', hatch=hatches[i % 4])
                ax.bar_label(bars, fmt='%.1f', fontsize=7)
            ax.set_xticks(range(len(prots)), [p.replace('_', ' ') for p in prots])
            ax.set(title=f'{dataset.upper()}: {"Rank-1" if metric == "R1" else "mAP"} (%)', ylim=(0, 105))
        axes[0].legend(fontsize=8)
        plt.tight_layout(); fig.savefig(f'{rc.OUT_DIR}/results_{dataset}.png', dpi=150); plt.show()


def save_grid(items, title, fname):
    """items: list of (image_path, caption) -> one row of images saved as PNG."""
    if not items:
        return
    fig, ax = plt.subplots(1, len(items), figsize=(1.45 * len(items), 3.4))
    for a, (p, cap) in zip(np.atleast_1d(ax), items):
        a.imshow(Image.open(p).convert('RGB').resize((64, 128))); a.set_title(cap, fontsize=7); a.axis('off')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(); fig.savefig(f'{rc.OUT_DIR}/{fname}', dpi=150, bbox_inches='tight'); plt.show()


def sample_images():
    random.seed(1)
    if ATTACHED['market']:
        ds = rc.load_dataset('market')
        by_id = collections.defaultdict(list)
        for r in ds.train:
            by_id[r[1]].append(r)
        pid = max(by_id, key=lambda p: len({r[2] for r in by_id[p]}))
        one_per_cam = sorted({r[2]: r for r in by_id[pid]}.values(), key=lambda r: r[2])
        save_grid([(r[0], f'cam {r[2] + 1}') for r in one_per_cam],
                  f'Market-1501: identity {pid} seen by {len(one_per_cam)} cameras', 'samples_market.png')
        junk = rc.parse_market(os.path.join(ds.root, 'bounding_box_test'), keep_junk=True)
        save_grid([(r[0], 'distractor\n(id 0)') for r in ds.sets['gallery'] if r[1] == 0][:5] +
                  [(r[0], 'junk\n(id -1)') for r in junk if r[1] == -1][:5],
                  'Market-1501: gallery distractors and junk images', 'samples_market_distractors.png')
    for name in ('prcc', 'ltcc'):
        if not ATTACHED[name]:
            continue
        ds = rc.load_dataset(name)
        by_id = collections.defaultdict(list)
        for r in ds.train:
            by_id[r[1]].append(r)
        pid = max(by_id, key=lambda p: (len({r[3] for r in by_id[p]}), len(by_id[p])))
        outfits = collections.defaultdict(list)
        for r in by_id[pid]:
            outfits[r[3]].append(r)
        items = [(r[0], f'outfit {k + 1}\ncam {r[2] + 1}') for k, rs in enumerate(list(outfits.values())[:4])
                 for r in random.sample(rs, min(3, len(rs)))]
        save_grid(items, f'{name.upper()}: identity {pid}, the same person in different clothes', f'samples_{name}.png')
        if name == 'prcc':     # the two PRCC test protocols for one test identity
            q = collections.defaultdict(lambda: collections.defaultdict(list))
            for s in ('gallery', 'query_same', 'query_diff'):
                for r in ds.sets[s]:
                    q[r[1]][s].append(r)
            tid = sorted(q)[0]
            items = [(r[0], cap) for s, cap in (('gallery', 'gallery\ncam A'), ('query_same', 'same-clothes\nquery, cam B'),
                                                 ('query_diff', 'clothes-changing\nquery, cam C'))
                     for r in q[tid][s][:3]]
            save_grid(items, f'PRCC test protocols (identity {tid})', 'samples_prcc_protocols.png')


try:
    rows = collect_results()
    if rows:
        with open(f'{rc.OUT_DIR}/results_summary.csv', 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"{'method':20s}{'dataset':8s}{'protocol':18s}{'R1':>7s}{'R5':>7s}{'R10':>7s}{'mAP':>7s}{'mINP':>7s}"
              f"{'paper R1/mAP':>15s}{'hours':>7s}")
        for r in rows:
            print(f"{r['method']:20s}{r['dataset']:8s}{r['protocol']:18s}{r['R1']:7.2f}{r['R5']:7.2f}{r['R10']:7.2f}"
                  f"{r['mAP']:7.2f}{r['mINP']:7.2f}{str(r['paper_R1']) + '/' + str(r['paper_mAP']):>15s}{r['train_hours']:7.2f}")
        with plt.style.context('grayscale'):
            results_charts(rows)
    else:
        print('No finished training results in outputs/ yet (attach earlier versions, see README).')
    sample_images()
except Exception:
    traceback.print_exc()
    open(f'{rc.OUT_DIR}/ERROR_cell7.txt', 'w').write(traceback.format_exc())


# ---- FAIR COMPARISON (PRCC, same protocol for all models, last epoch, every seed found in outputs/)
FAIR_METHODS = [('baseline', 'Baseline (ResNet-50)'), ('transreid', 'TransReID (ViT-B/16)'), ('cal', 'CAL (ResNet-50)')]
FAIR_PROTOCOLS = [('clothes_changing', 'CC'), ('same_clothes', 'SC')]


def fair_summary():
    rows, per_method = [], {}
    for key, label in FAIR_METHODS:
        runs = []
        for path in sorted(glob.glob(f'{rc.OUT_DIR}/{tag("fair_" + key)}_s*_history.json')):
            h = json.load(open(path))
            if not h['history'] or h['history'][-1]['epoch'] < FAIR['epochs'] and not QUICK:
                print(f'   {os.path.basename(path)}: unfinished, ignored'); continue
            last = [r for r in h['history'] if 'mAP' in r][-1]           # last evaluated epoch = reported model
            seed = int(re.search(r'_s(\d+)_history', path).group(1))
            runs.append(dict(seed=seed, epoch=last['epoch'], hours=sum(r['time'] for r in h['history']) / 3600,
                             **{f'{p}/{m}': last[f'{p}/{m}'] * 100 for p, _ in FAIR_PROTOCOLS
                                for m in ('R1', 'R5', 'R10', 'mAP', 'mINP')}))
        if not runs:
            continue
        per_method[label] = runs
        row = dict(method=label, seeds=len(runs), epoch=runs[0]['epoch'],
                   train_hours=round(np.mean([r['hours'] for r in runs]), 2))
        for p, short in FAIR_PROTOCOLS:
            for m in ('R1', 'R5', 'R10', 'mAP', 'mINP'):
                v = [r[f'{p}/{m}'] for r in runs]
                row[f'{short}_{m}'] = round(float(np.mean(v)), 2)
                row[f'{short}_{m}_std'] = round(float(np.std(v)), 2) if len(v) > 1 else ''
        rows.append(row)
    if not rows:
        print('\nFAIR COMPARISON: no finished fair_* runs in outputs/ yet.'); return
    with open(f'{rc.OUT_DIR}/fair_comparison_prcc.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print('\nFAIR COMPARISON on PRCC (same protocol, last epoch; mean +- std over seeds)')
    print(f"{'method':24s}{'seeds':>6s}{'CC R1':>14s}{'CC mAP':>14s}{'CC mINP':>14s}{'SC R1':>9s}{'SC mAP':>9s}{'hours':>7s}")
    pm = lambda r, k: f"{r[k]:.2f}" + (f"+-{r[k + '_std']:.2f}" if r[k + '_std'] != '' else '')
    for r in rows:
        print(f"{r['method']:24s}{r['seeds']:6d}{pm(r, 'CC_R1'):>14s}{pm(r, 'CC_mAP'):>14s}{pm(r, 'CC_mINP'):>14s}"
              f"{r['SC_R1']:9.2f}{r['SC_mAP']:9.2f}{r['train_hours']:7.2f}")
    with plt.style.context('grayscale'):
        # bar chart: clothes-changing R1 / mAP / mINP per method (error bars = std over seeds)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        metrics, w = ('R1', 'mAP', 'mINP'), 0.26
        for i, m in enumerate(metrics):
            vals = [r[f'CC_{m}'] for r in rows]
            err = [r[f'CC_{m}_std'] or 0 for r in rows]
            b = ax.bar(np.arange(len(rows)) + (i - 1) * w, vals, w, yerr=err, capsize=3, label=m,
                       color=('0.3', 'white', '0.7')[i], edgecolor='black', hatch=('', '//', '')[i])
            ax.bar_label(b, fmt='%.1f', fontsize=7, padding=2)
        ax.set_xticks(range(len(rows)), [r['method'] for r in rows], fontsize=8)
        ax.set(ylabel='%', ylim=(0, 70), title='PRCC clothes-changing, fair protocol (last epoch)')
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout(); fig.savefig(f'{rc.OUT_DIR}/fair_comparison_prcc.png', dpi=150); plt.show()
        # curves: clothes-changing mAP during training (seed with the lowest number)
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        for (key, label), style in zip(FAIR_METHODS, ('s--', '^:', 'o-')):
            files = sorted(glob.glob(f'{rc.OUT_DIR}/{tag("fair_" + key)}_s*_history.json'))
            if not files:
                continue
            ev = [r for r in json.load(open(files[0]))['history'] if 'mAP' in r]
            ax.plot([r['epoch'] for r in ev], [r['clothes_changing/mAP'] * 100 for r in ev], style, color='black', label=label)
        ax.axvline(25.5, color='0.6', lw=1); ax.text(26.5, ax.get_ylim()[0] + 1, 'CAL loss on', fontsize=8, color='0.3')
        ax.set(xlabel='epoch', ylabel='CC mAP (%)', title='PRCC clothes-changing mAP during training (fair protocol)')
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout(); fig.savefig(f'{rc.OUT_DIR}/fair_curves_prcc.png', dpi=150); plt.show()


try:
    fair_summary()
except Exception:
    traceback.print_exc()
    open(f'{rc.OUT_DIR}/ERROR_cell7_fair.txt', 'w').write(traceback.format_exc())


# ===== CELL 8 : REPORT 2  retrieval examples (query + top-10 gallery matches) from the trained models =====
# Uses <job>_best.pth. The SAME queries are shown for every method, so the figures can be compared directly.
# Solid thick frame + tick = correct person; dashed thin frame + cross = wrong person.
RETRIEVAL = {'transreid_market': 'standard', 'transreid_prcc': 'clothes_changing', 'cal_prcc': 'clothes_changing',
             'baseline_prcc': 'clothes_changing', 'transreid_ltcc': 'clothes_changing',
             'cal_ltcc': 'clothes_changing', 'baseline_ltcc': 'clothes_changing',
             f'fair_baseline_s{SEED}': 'clothes_changing', f'fair_transreid_s{SEED}': 'clothes_changing',
             f'fair_cal_s{SEED}': 'clothes_changing'}
N_QUERIES, TOP_K = 5, 10


def load_trained(job):
    if job.startswith('fair_'):   # fair_<method>_s<seed>: FAIR test pre-processing, flip for every model
        method, dataset = job.split('_')[1], 'prcc'
        if method == 'transreid':
            data = rc.build_data(dataset, size=FAIR['size'], mean=[0.5] * 3, std=[0.5] * 3, interp=FAIR['interp'])
            model, fwd = TransReID(data.num_classes, data.num_cams), (lambda m, x, c: m(x, c))
        else:
            data = rc.build_data(dataset, size=FAIR['size'], interp=FAIR['interp'])
            model, fwd = CALNet(data.num_classes, data.num_clothes), None
        model.load_state_dict(torch.load(f'{rc.OUT_DIR}/{tag(job)}_best.pth', map_location='cpu'))
        return model.to(rc.DEVICE).eval(), data, fwd, FAIR['flip_test']
    method, dataset = job.split('_')
    if method == 'transreid':     # same test pre-processing as run_transreid
        data = rc.build_data(dataset, size=(256, 128), mean=[0.5] * 3, std=[0.5] * 3, interp='bicubic')
        model, fwd, flip = TransReID(data.num_classes, data.num_cams), (lambda m, x, c: m(x, c)), False
    else:                         # same test pre-processing as run_cal
        data = rc.build_data(dataset, size=(384, 192), batch_size=32, num_instances=8, crop='cal')
        model, fwd, flip = CALNet(data.num_classes, data.num_clothes), None, True
    model.load_state_dict(torch.load(f'{rc.OUT_DIR}/{tag(job)}_best.pth', map_location='cpu'))
    return model.to(rc.DEVICE).eval(), data, fwd, flip


def retrieval_figure(job, protocol):
    model, data, fwd, flip = load_trained(job)
    ql, gl, mode = data.tests[protocol]
    qf, qp, qcam = rc.extract_features(model, ql, flip, fwd)
    gf, gp, gcam = rc.extract_features(model, gl, flip, fwd)
    dist = rc.cosine_distmat(qf, gf)
    qclo, gclo = ql.dataset.clothes, gl.dataset.clothes
    queries = np.random.default_rng(0).choice(len(qp), min(N_QUERIES, len(qp)), replace=False)   # same for every method
    fig, axes = plt.subplots(len(queries), TOP_K + 1, figsize=(1.25 * (TOP_K + 1), 2.6 * len(queries)), squeeze=False)
    for row, q in zip(axes, queries):
        order = np.argsort(dist[q])
        same = gp[order] == qp[q]
        junk = same & (gcam[order] == qcam[q])
        if mode == 'cc':
            junk |= same & (gclo[order] == qclo[q])
        order, same = order[~junk][:TOP_K], same[~junk][:TOP_K]
        cells = [(ql.dataset.data[q][0], 'query', None)] + \
                [(gl.dataset.data[g][0], f'{k + 1} ' + ('✓' if ok else '✗'), bool(ok))
                 for k, (g, ok) in enumerate(zip(order, same))]
        for a in row:
            a.axis('off')
        for a, (p, cap, ok) in zip(row, cells):
            a.axis('on')
            a.imshow(Image.open(p).convert('RGB').resize((64, 128))); a.set_title(cap, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(1 if ok is False else 3); s.set_linestyle('--' if ok is False else '-')
    method, dataset = (job.split('_')[1], 'prcc') if job.startswith('fair_') else job.split('_')
    extra = ', fair protocol' if job.startswith('fair_') else ''
    fig.suptitle(f'{METHOD_NAME[method]} on {dataset.upper()} ({protocol.replace("_", " ")}{extra}): top-{TOP_K} matches', fontsize=12)
    plt.tight_layout(); fig.savefig(f'{rc.OUT_DIR}/retrieval_{tag(job)}.png', dpi=130, bbox_inches='tight'); plt.show()
    del model; gc.collect(); torch.cuda.empty_cache()


for job, protocol in RETRIEVAL.items():
    if not os.path.exists(f'{rc.OUT_DIR}/{tag(job)}_best.pth'):
        print(f'{job}: no trained model in outputs/, skipped'); continue
    print(f'\n=== retrieval examples: {job}')
    try:
        retrieval_figure(job, protocol)
    except Exception:
        traceback.print_exc()
        open(f'{rc.OUT_DIR}/ERROR_cell8_{job}.txt', 'w').write(traceback.format_exc())
print('\nReport files in outputs/:', sorted(f for f in os.listdir(rc.OUT_DIR) if f.endswith(('.png', '.csv', '.json'))))


# ===== CELL 9 : ONE-CLICK DOWNLOAD  zip of all report files (tables, graphs, images, logs) =====
# Creates /kaggle/working/report_files.zip and shows a download link under this cell.
# Model weights (.pth) are left out because they are large; set INCLUDE_WEIGHTS = True to add *_best.pth.
import zipfile
from IPython.display import FileLink, display

INCLUDE_WEIGHTS = False
zip_path = '/kaggle/working/report_files.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(rc.OUT_DIR)):
        if f.endswith(('_ckpt.pth', '.tmp')) or (f.endswith('.pth') and not (INCLUDE_WEIGHTS and f.endswith('_best.pth'))):
            continue
        z.write(os.path.join(rc.OUT_DIR, f), arcname=f'outputs/{f}')
        print('  added', f)
print(f'{zip_path}: {os.path.getsize(zip_path) / 1e6:.1f} MB')
os.chdir('/kaggle/working')
display(FileLink('report_files.zip'))
