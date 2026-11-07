%%writefile /kaggle/working/reid_common.py
"""CELL 1 (shared code): dataset, samplers, losses, training loop, evaluation. Paste down to the CELL 2 marker."""
import os, re, glob, copy, json, math, time, random
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
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================ data
def find_dataset_root(hint='market', base='/kaggle/input'):
    """Find the folder that holds bounding_box_train / bounding_box_test / query (Market-1501 layout)."""
    need = {'bounding_box_train', 'bounding_box_test', 'query'}
    found = []
    for root, dirs, _ in os.walk(base):
        if need <= set(dirs):
            found.append(root); dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith(('bounding_box', 'gt_', 'query'))]
    if not found:
        raise FileNotFoundError(f'No folder containing {need} under {base}. Attach the dataset with "+ Add Input".')
    hits = [r for r in found if hint.lower() in r.lower()] or found
    print('Dataset root:', hits[0])
    return hits[0]


PID_RE = re.compile(r'([-\d]+)_c(\d+)')   # Market: 0002_c1s1_000451_03.jpg   Duke: 0001_c2_f0046182.jpg


def parse_dir(folder, relabel=False, keep_junk=False):
    """Return list of (path, person_id, camera_id[0-based]). id -1 = junk, id 0 = distractor."""
    data = []
    for p in sorted(glob.glob(os.path.join(folder, '*.jpg'))):
        m = PID_RE.search(os.path.basename(p))
        if m is None:
            continue
        pid, cam = int(m.group(1)), int(m.group(2)) - 1
        if pid == -1 and not keep_junk:
            continue
        data.append((p, pid, cam))
    if relabel:
        lut = {pid: i for i, pid in enumerate(sorted({d[1] for d in data}))}
        data = [(p, lut[pid], c) for p, pid, c in data]
    return data


class ReIDDataset(Dataset):
    def __init__(self, data, transform):
        self.data, self.transform = data, transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, pid, cam = self.data[i]
        return self.transform(Image.open(path).convert('RGB')), pid, cam, i


def build_transforms(size, train, mean=IMAGENET_MEAN, std=IMAGENET_STD, crop=True, erasing=0.5, interp='bilinear'):
    mode = T.InterpolationMode.BICUBIC if interp == 'bicubic' else T.InterpolationMode.BILINEAR
    t = [T.Resize(size, interpolation=mode)]
    if train:
        t.append(T.RandomHorizontalFlip())
        if crop:
            t += [T.Pad(10), T.RandomCrop(size)]
    t += [T.ToTensor(), T.Normalize(mean, std)]
    if train and erasing > 0:
        t.append(T.RandomErasing(p=erasing, scale=(0.02, 0.4), value=0))   # 0 = mean pixel after Normalize
    return T.Compose(t)


class RandomIdentitySampler(Sampler):
    """P identities x K instances per batch (needed by triplet loss)."""
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


def build_data(root, size=(256, 128), batch_size=64, sampler='pk', num_instances=4, workers=4,
               mean=IMAGENET_MEAN, std=IMAGENET_STD, crop=True, erasing=0.5, interp='bilinear'):
    train = parse_dir(os.path.join(root, 'bounding_box_train'), relabel=True)
    query = parse_dir(os.path.join(root, 'query'))
    gallery = parse_dir(os.path.join(root, 'bounding_box_test'))
    tr_tf = build_transforms(size, True, mean, std, crop, erasing, interp)
    te_tf = build_transforms(size, False, mean, std, interp=interp)
    kw = dict(num_workers=workers, pin_memory=True)
    if sampler == 'pk':
        train_loader = DataLoader(ReIDDataset(train, tr_tf), batch_size, drop_last=True,
                                  sampler=RandomIdentitySampler(train, batch_size, num_instances), **kw)
    else:
        train_loader = DataLoader(ReIDDataset(train, tr_tf), batch_size, shuffle=True, drop_last=True, **kw)
    d = SimpleNamespace(
        root=root, train=train, query=query, gallery=gallery, train_loader=train_loader,
        query_loader=DataLoader(ReIDDataset(query, te_tf), 256, shuffle=False, **kw),
        gallery_loader=DataLoader(ReIDDataset(gallery, te_tf), 256, shuffle=False, **kw),
        num_classes=len({x[1] for x in train}), num_cams=1 + max(x[2] for x in train + query + gallery))
    print(f'train {len(train)} imgs / {d.num_classes} ids | query {len(query)} imgs / {len({x[1] for x in query})} ids | '
          f'gallery {len(gallery)} imgs / {len({x[1] for x in gallery})} ids | cameras {d.num_cams}')
    return d


# ============================================================ losses
def euclidean_dist(x, y):
    xx, yy = x.pow(2).sum(1, keepdim=True), y.pow(2).sum(1, keepdim=True).t()
    return (xx + yy - 2 * x @ y.t()).clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """Batch-hard triplet loss (Hermans et al. 2017)."""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels):
        d = euclidean_dist(feats.float(), feats.float())
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        d_ap = d.masked_fill(~same, -1).max(1)[0]              # hardest positive
        d_an = d.masked_fill(same, float('inf')).min(1)[0]     # hardest negative
        return F.margin_ranking_loss(d_an, d_ap, torch.ones_like(d_an), margin=self.margin)


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


def evaluate_rank(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=50):
    """Market-1501 protocol: gallery images with same id AND same camera as the query are ignored."""
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None])
    all_cmc, all_ap, all_inp = [], [], []
    for q in range(len(q_pids)):
        order = indices[q]
        keep = ~((g_pids[order] == q_pids[q]) & (g_cams[order] == q_cams[q]))
        raw = matches[q][keep].astype(np.float32)
        if not raw.any():
            continue
        cum = raw.cumsum()
        all_inp.append(raw.sum() / (np.where(raw == 1)[0].max() + 1.0))
        all_cmc.append(np.minimum(cum, 1)[:max_rank])
        all_ap.append(((cum / (np.arange(len(raw)) + 1.0)) * raw).sum() / raw.sum())
    return np.mean(all_cmc, 0), float(np.mean(all_ap)), float(np.mean(all_inp))


def test(model, data, flip=False, forward_fn=None):
    t0 = time.time()
    qf, qp, qc = extract_features(model, data.query_loader, flip, forward_fn)
    gf, gp, gc = extract_features(model, data.gallery_loader, flip, forward_fn)
    cmc, mAP, mINP = evaluate_rank(cosine_distmat(qf, gf), qp, gp, qc, gc)
    res = {'mAP': mAP, 'R1': float(cmc[0]), 'R5': float(cmc[4]), 'R10': float(cmc[9]), 'mINP': mINP}
    print('   TEST  ' + '  '.join(f'{k}={v * 100:.2f}' for k, v in res.items()) + f'  ({time.time() - t0:.0f}s)')
    return res


# ============================================================ training helpers
def autocast():
    return torch.autocast('cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda')


def make_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')


def warmup_multistep(optimizer, milestones, gamma=0.1, warmup_epochs=10, warmup_factor=0.01):
    def f(e):
        w = 1.0 if e >= warmup_epochs else warmup_factor + (1 - warmup_factor) * e / warmup_epochs
        return w * gamma ** sum(e >= m for m in milestones)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def warmup_cosine(optimizer, total_epochs, warmup_epochs=5, warmup_factor=0.01, min_ratio=0.002):
    def f(e):
        if e < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * e / warmup_epochs
        p = (e - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def load_matching(model, path):
    """Load every tensor whose name and shape match (skips classifier etc.)."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    own.update(ok); model.load_state_dict(own)
    print(f'Loaded {len(ok)}/{len(own)} tensors from {path}')


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
        dirs[:] = [d for d in dirs if not d.startswith(('bounding_box', 'gt_', 'query'))]
        if os.path.basename(root) != 'outputs' or not any(f.endswith(('_ckpt.pth', '_history.json')) for f in files):
            continue
        for f in files:
            dst = os.path.join(OUT_DIR, f)
            if not f.startswith('quick_') and not f.endswith('.tmp') and not os.path.exists(dst):
                shutil.copy2(os.path.join(root, f), dst); n += 1
                print('   restored', f, 'from', root)
    print(f'restored {n} file(s) from previous runs' if n else 'no previous outputs attached (fresh start)')


def fit(model, data, optimizer, scheduler, step_fn, epochs, name, eval_every=10,
        flip=False, test_forward=None, start_hook=None):
    """Generic epoch loop. step_fn(model, imgs, pids, cams, idx) -> loss.
    Saves a full checkpoint every epoch, resumes from it automatically, and stops cleanly before the time limit."""
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
        rec = {'epoch': ep + 1, 'loss': tot / max(n, 1), 'lr': optimizer.param_groups[0]['lr'], 'time': time.time() - t0}
        scheduler.step()
        print(f"[{name}] epoch {ep + 1}/{epochs}  loss {rec['loss']:.4f}  lr {rec['lr']:.2e}  {rec['time']:.0f}s")
        if (ep + 1) % eval_every == 0 or ep + 1 == epochs:
            t1 = time.time()
            rec.update(test(model, data, flip, test_forward))
            eval_time = time.time() - t1
            if rec['mAP'] > best['mAP']:
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


def plot_history(history, name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot([h['epoch'] for h in history], [h['loss'] for h in history])
    ax[0].set(title=f'{name}: training loss', xlabel='epoch')
    ev = [h for h in history if 'mAP' in h]
    for k in ('mAP', 'R1'):
        ax[1].plot([h['epoch'] for h in ev], [h[k] * 100 for h in ev], 'o-', label=k)
    ax[1].set(title=f'{name}: test accuracy', xlabel='epoch', ylabel='%'); ax[1].legend()
    plt.tight_layout(); plt.savefig(f'{OUT_DIR}/{name}_curves.png', dpi=120); plt.show()


def compare(rows):
    """rows: list of (method, ours_dict, paper_R1, paper_mAP) -> prints a table for the report."""
    print(f"{'method':22s}{'R1':>8s}{'mAP':>8s}{'mINP':>8s}   |  paper R1 / mAP")
    for name, r, p_r1, p_map in rows:
        print(f"{name:22s}{r['R1'] * 100:8.2f}{r['mAP'] * 100:8.2f}{r['mINP'] * 100:8.2f}   |  {p_r1} / {p_map}")


# ===== CELL 2 : SWITCH (the only cell you edit) =====
PAPER = 'all'      # 'explore' | 'pcb' | 'osnet' | 'transreid' | 'spcl' | 'all'
QUICK = True       # True  = smoke test, 1 short epoch per paper (all papers together ~25 min)
                   # False = real training run (one paper per "Save & Run All", see README)
MAX_HOURS = 11.5   # training stops cleanly (checkpoint saved) before this; Kaggle's hard limit is 12 h

import sys, os, gc, copy, math, json, time, glob, random, subprocess, importlib, collections
sys.path.insert(0, '/kaggle/working')
import numpy as np, scipy.sparse as sp, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torch.utils.data import DataLoader, Sampler
from PIL import Image
from sklearn.cluster import DBSCAN
import reid_common as rc; importlib.reload(rc)

rc.set_seed(0)
rc.MAX_HOURS = MAX_HOURS
ROOT = rc.find_dataset_root('market')
tag = lambda name: ('quick_' if QUICK else '') + name          # quick-test outputs never overwrite real ones
if QUICK:                                                      # smoke tests always start from scratch
    for f in glob.glob(f'{rc.OUT_DIR}/quick_*'):
        os.remove(f)
else:                                                          # pick up checkpoints of an attached earlier version
    rc.restore_previous_outputs()
print(f'PAPER={PAPER}  QUICK={QUICK}  torch {torch.__version__}  GPUs: {torch.cuda.device_count()}',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')


# ===== CELL 3 : DATASET EXPLORATION =====
def run_explore():
    splits = {name: rc.parse_dir(os.path.join(ROOT, d), keep_junk=True)
              for name, d in [('train', 'bounding_box_train'), ('query', 'query'), ('gallery', 'bounding_box_test')]}
    stats = {}
    print(f"{'split':9s}{'images':>8s}{'IDs':>6s}{'distractors(id 0)':>19s}{'junk(id -1)':>13s}{'cams':>6s}")
    for name, d in splits.items():
        pids = np.array([x[1] for x in d])
        s = stats[name] = dict(images=len(d), ids=len(set(pids[pids > 0])), distractors=int((pids == 0).sum()),
                               junk=int((pids == -1).sum()), cams=len({x[2] for x in d}))
        print(f"{name:9s}{s['images']:8d}{s['ids']:6d}{s['distractors']:19d}{s['junk']:13d}{s['cams']:6d}")
    ids = {k: {x[1] for x in v if x[1] > 0} for k, v in splits.items()}
    print('\ntrain/test identity overlap :', len(ids['train'] & ids['gallery']), '(must be 0)')
    print('query IDs all in gallery    :', ids['query'] <= ids['gallery'])
    print('filename example            :', os.path.basename(splits['train'][0][0]),
          ' -> <person id>_c<camera>s<sequence>_<frame>_<bbox no.>.jpg')
    sizes = collections.Counter(Image.open(p).size for p, _, _ in random.sample(splits['train'], 500))
    print('image sizes (W,H), 500 samples:', sizes.most_common(3))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    cnt = collections.Counter(x[1] for x in splits['train'])
    ax[0].hist(list(cnt.values()), bins=40)
    ax[0].set(title=f'train: images per identity (mean {np.mean(list(cnt.values())):.1f})', xlabel='#images', ylabel='#identities')
    for i, (name, d) in enumerate(splits.items()):
        c = collections.Counter(x[2] + 1 for x in d)
        cams = sorted(c)
        ax[1].bar(np.array(cams) + (i - 1) * 0.27, [c[k] for k in cams], width=0.27, label=name)
    ax[1].set(title='images per camera', xlabel='camera'); ax[1].legend()
    cam_per_id = collections.defaultdict(set)
    for _, p, c in splits['train']:
        cam_per_id[p].add(c)
    ax[2].hist([len(v) for v in cam_per_id.values()], bins=np.arange(1, 10) - 0.5, rwidth=0.8)
    ax[2].set(title='train: cameras per identity', xlabel='#cameras')
    plt.tight_layout(); plt.savefig(f'{rc.OUT_DIR}/market_stats.png', dpi=120); plt.show()

    def show(items, title, n=12):
        items = items[:n]
        if not items:
            return
        fig, ax = plt.subplots(1, len(items), figsize=(1.5 * len(items), 3.3))
        for a, (p, pid, cam) in zip(np.atleast_1d(ax), items):
            a.imshow(Image.open(p)); a.set_title(f'id {pid}\ncam {cam + 1}', fontsize=8); a.axis('off')
        fig.suptitle(title); plt.show()

    random.seed(1)
    by_id = collections.defaultdict(list)
    for x in splits['train']:
        by_id[x[1]].append(x)
    pid = random.choice([p for p, v in by_id.items() if len({c for *_, c in v}) >= 4])
    one_per_cam = sorted({x[2]: x for x in by_id[pid]}.values(), key=lambda x: x[2])
    show(one_per_cam + random.sample(by_id[pid], min(4, len(by_id[pid]))), f'train identity {pid}: same person, different cameras')
    q = random.choice(splits['query'])
    pos = [x for x in splits['gallery'] if x[1] == q[1]]
    show([q] + pos, f'query (1st) + its {len(pos)} gallery images (same-camera matches are ignored at test time)')
    show([x for x in splits['gallery'] if x[1] == 0][:6] + [x for x in splits['gallery'] if x[1] == -1][:6],
         'gallery distractors (id 0) and junk (id -1)')
    json.dump(stats, open(f'{rc.OUT_DIR}/market_stats.json', 'w'), indent=1)


# ===== CELL 4 : PAPER 1  PCB + RPP (Sun et al., ECCV 2018) =====
class PCB(nn.Module):
    """ResNet-50 (last stride 1) -> 24x8 map -> 6 horizontal stripes -> 6 independent ID classifiers.
    RPP replaces the uniform stripes with a learned per-pixel part classifier (soft re-assignment)."""
    def __init__(self, num_classes, parts=6, reduce_dim=256):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.parts, self.rpp = parts, False
        self.part_classifier = nn.Conv2d(2048, parts, 1)
        self.reduce = nn.ModuleList([nn.Sequential(nn.Conv2d(2048, reduce_dim, 1, bias=False),
                                                   nn.BatchNorm2d(reduce_dim), nn.ReLU(inplace=True)) for _ in range(parts)])
        self.dropout = nn.Dropout(0.5)
        self.classifiers = nn.ModuleList([nn.Linear(reduce_dim, num_classes) for _ in range(parts)])
        for c in self.classifiers:
            nn.init.normal_(c.weight, std=0.001); nn.init.zeros_(c.bias)

    def pool(self, x):                                       # -> B x 2048 x parts x 1
        if not self.rpp:
            return F.adaptive_avg_pool2d(x, (self.parts, 1))  # uniform partition
        w = F.softmax(self.part_classifier(x), dim=1)        # B x parts x H x W, P(part | pixel)
        g = torch.einsum('bphw,bchw->bcp', w.to(x.dtype), x) / (w.sum((2, 3)).unsqueeze(1).to(x.dtype) + 1e-6)
        return g.unsqueeze(-1)

    def forward(self, x):
        g = self.pool(self.backbone(x))
        if not self.training:
            return g.flatten(1)                              # 6 x 2048 descriptor used at test time
        return [cls(self.dropout(red(g[:, :, i:i + 1]).flatten(1)))
                for i, (red, cls) in enumerate(zip(self.reduce, self.classifiers))]


def pcb_step(model, imgs, pids, cams, idx):
    return sum(F.cross_entropy(logit.float(), pids) for logit in model(imgs))


def pcb_param_groups(model, lr_backbone, lr_head):
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    return [{'params': list(model.backbone.parameters()), 'lr': lr_backbone}, {'params': head, 'lr': lr_head}]


def run_pcb():
    # Paper setting: 384x128 input, horizontal flip only, batch 64, plain shuffling
    data = rc.build_data(ROOT, size=(384, 128), batch_size=64, sampler='random', crop=False, erasing=0.0)
    E1, E_RPP = (1, 1) if QUICK else (60, 10)
    sgd = lambda params: torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # step 1: PCB, 60 epochs, lr x0.1 at epoch 40
    model = PCB(data.num_classes).to(rc.DEVICE)
    model.part_classifier.requires_grad_(False)
    opt = sgd(pcb_param_groups(model, 0.01, 0.1))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40], gamma=0.1)
    hist_pcb, best_pcb = rc.fit(model, data, opt, sched, pcb_step, E1, tag('pcb'), eval_every=10)
    if rc.STOPPED:
        return

    # step 2: RPP, train ONLY the part classifier
    model.rpp = True
    nn.init.normal_(model.part_classifier.weight, std=0.001); nn.init.zeros_(model.part_classifier.bias)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith('part_classifier')
    opt = sgd([{'params': list(model.part_classifier.parameters()), 'lr': 0.01}])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000])
    rc.fit(model, data, opt, sched, pcb_step, E_RPP, tag('pcb_rpp_step2'), eval_every=10)
    if rc.STOPPED:
        return

    # step 3: fine-tune the whole network
    for p in model.parameters():
        p.requires_grad = True
    opt = sgd(pcb_param_groups(model, 0.001, 0.01))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000])
    hist_rpp, best_rpp = rc.fit(model, data, opt, sched, pcb_step, E_RPP, tag('pcb_rpp'), eval_every=5)
    if rc.STOPPED:
        return

    rc.plot_history(hist_pcb, tag('pcb'))
    rc.compare([('PCB', best_pcb, 92.3, 77.4), ('PCB+RPP', best_rpp, 93.8, 81.6)])


# ===== CELL 5 : PAPER 2  OSNet (Zhou et al., ICCV 2019) =====
# Layer names follow torchreid so the official ImageNet weights load.
class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn, self.relu = nn.BatchNorm2d(cout), nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    def __init__(self, cin, cout, stride=1):
        super().__init__(cin, cout, 1, stride)


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 pointwise + 3x3 depthwise (the 'Lite 3x3' unit)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn, self.relu = nn.BatchNorm2d(cout), nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Unified aggregation gate (shared by all streams): channel-wise weights from a tiny MLP."""
    def __init__(self, c, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c // reduction, c, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        return x * self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))


class OSBlock(nn.Module):
    """Omni-scale residual block: 4 streams with 1..4 stacked Lite3x3 (receptive fields 3,5,7,9) fused by the gate."""
    def __init__(self, cin, cout, reduction=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = sum(self.gate(conv(x1)) for conv in (self.conv2a, self.conv2b, self.conv2c, self.conv2d))
        identity = x if self.downsample is None else self.downsample(x)
        return F.relu(self.conv3(x2) + identity)


class OSNet(nn.Module):
    def __init__(self, num_classes, layers=(2, 2, 2), channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(feature_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    @staticmethod
    def _make_layer(n, cin, cout, reduce_spatial):
        layers = [OSBlock(cin, cout)] + [OSBlock(cout, cout) for _ in range(n - 1)]
        if reduce_spatial:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        v = self.fc(self.global_avgpool(x).flatten(1))
        return v if not self.training else self.classifier(v)


def get_osnet_weights():
    """Official torchreid ImageNet weights: Kaggle input if uploaded, else download from Google Drive."""
    found = glob.glob('/kaggle/input/*/osnet_x1_0_imagenet.pth') + glob.glob('/kaggle/input/*/*/osnet_x1_0_imagenet.pth')
    W = found[0] if found else '/kaggle/working/osnet_x1_0_imagenet.pth'
    if not os.path.exists(W):
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=False)
            import gdown
            gdown.download('https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY', W, quiet=False)
        except Exception as e:
            print('download failed:', e)
    return W if os.path.exists(W) else None


def osnet_step(model, imgs, pids, cams, idx):
    return F.cross_entropy(model(imgs).float(), pids, label_smoothing=0.1)


def run_osnet():
    # torchreid recipe: 256x128, random flip, batch 64, AMSGrad, cosine LR, label-smoothed CE,
    # first 10 epochs train the classifier only. Paper uses 250 epochs; 150 fits Kaggle time.
    data = rc.build_data(ROOT, size=(256, 128), batch_size=64, sampler='random', crop=False, erasing=0.0)
    EPOCHS, FIXBASE = (2, 1) if QUICK else (150, 10)
    model = OSNet(data.num_classes)
    W = get_osnet_weights()
    if W:
        rc.load_matching(model, W)
    else:
        print('!! ImageNet weights not found, training from scratch (expect much lower accuracy)')
    print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')
    opt = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=5e-4, betas=(0.9, 0.99), amsgrad=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    def fixbase(model, ep):
        for n, p in model.named_parameters():
            p.requires_grad = ep >= FIXBASE or n.startswith('classifier')

    hist, best = rc.fit(model, data, opt, sched, osnet_step, EPOCHS, tag('osnet'), eval_every=10, start_hook=fixbase)
    if rc.STOPPED:
        return
    rc.plot_history(hist, tag('osnet'))
    rc.compare([('OSNet x1.0', best, 94.8, 84.9)])


# ===== CELL 6 : PAPER 3  TransReID (He et al., ICCV 2021) =====
# ViT-B/16 + overlapping patches (stride 12) + SIE (camera embedding) + JPM (jigsaw patch module).
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
        try:
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
        if not self.training:
            return torch.cat([feats[0]] + [f / self.divide for f in feats[1:]], 1)
        logits = [c(bn(f)) for f, bn, c in zip(feats, self.bnnecks, self.classifiers)]
        return logits, feats


triplet = rc.TripletLoss(margin=0.3)


def transreid_step(model, imgs, pids, cams, idx):
    logits, feats = model(imgs, cams)
    ce = [F.cross_entropy(l.float(), pids) for l in logits]
    tri = [triplet(f, pids) for f in feats]
    if len(ce) == 1:
        return ce[0] + tri[0]
    # global branch weight 0.5, mean of the JPM local branches weight 0.5
    return 0.5 * (ce[0] + tri[0]) + 0.5 * (sum(ce[1:]) + sum(tri[1:])) / (len(ce) - 1)


def run_transreid():
    # Paper setting: 256x128, flip + pad/crop + random erasing, 16 ids x 4 imgs, mean/std 0.5,
    # SGD 0.008, 5 warm-up epochs + cosine, 120 epochs (use 60 if Kaggle time is short)
    data = rc.build_data(ROOT, size=(256, 128), batch_size=64, sampler='pk', num_instances=4,
                         mean=[0.5] * 3, std=[0.5] * 3, crop=True, erasing=0.5, interp='bicubic')
    EPOCHS, BASE_LR = (1 if QUICK else 120), 0.008
    model = TransReID(data.num_classes, data.num_cams)
    groups = [{'params': [p], 'lr': BASE_LR * (2 if 'bias' in n else 1), 'weight_decay': 1e-4}
              for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD(groups, lr=BASE_LR, momentum=0.9)
    sched = rc.warmup_cosine(opt, EPOCHS, warmup_epochs=5)
    model = model.to(rc.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)                     # uses both T4 GPUs
    hist, best = rc.fit(model, data, opt, sched, transreid_step, EPOCHS, tag('transreid'), eval_every=20,
                        test_forward=lambda m, x, c: m(x, c))
    if rc.STOPPED:
        return
    rc.plot_history(hist, tag('transreid'))
    rc.compare([('TransReID* (s=12)', best, 95.2, 88.9)])


# ===== CELL 7 : PAPER 4  SpCL (Ge et al., NeurIPS 2020), fully UNSUPERVISED =====
# Each epoch: k-reciprocal Jaccard distance -> DBSCAN pseudo labels -> self-paced reliability filter -> 400 iters.
class SpCLNet(nn.Module):
    """ResNet-50 (last stride 1) -> GAP -> BN -> L2-normalised 2048-d feature."""
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(r.children())[:-2])
        self.feat_bn = nn.BatchNorm1d(2048)
        nn.init.ones_(self.feat_bn.weight); nn.init.zeros_(self.feat_bn.bias)
        self.feat_bn.bias.requires_grad_(False)

    def forward(self, x):
        return F.normalize(self.feat_bn(F.adaptive_avg_pool2d(self.base(x), 1).flatten(1)), dim=1)


class HM(torch.autograd.Function):
    """inputs @ memory^T; in backward, momentum-update the memory slots of the batch instances."""
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features, ctx.momentum = features, momentum
        ctx.save_for_backward(inputs, indexes)
        return inputs.mm(features.t())

    @staticmethod
    def backward(ctx, grad_out):
        inputs, indexes = ctx.saved_tensors
        grad_in = grad_out.mm(ctx.features) if ctx.needs_input_grad[0] else None
        for x, y in zip(inputs, indexes.tolist()):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        return grad_in, None, None, None


class HybridMemory(nn.Module):
    """Unified contrastive loss: positive = own cluster centroid (or own instance if un-clustered),
    negatives = all other cluster centroids + all un-clustered instances."""
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super().__init__()
        self.temp, self.momentum = temp, momentum
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples, dtype=torch.long))

    def forward(self, inputs, indexes):
        inputs = F.normalize(inputs.float(), dim=1)
        logits = HM.apply(inputs, indexes, self.features, self.momentum) / self.temp       # B x N
        C, dev = int(self.labels.max()) + 1, logits.device
        sim = torch.zeros(C, logits.size(0), device=dev).index_add_(0, self.labels, logits.t().contiguous())
        nums = torch.zeros(C, 1, device=dev).index_add_(0, self.labels, torch.ones(len(self.labels), 1, device=dev))
        sim = (sim / nums.clamp(min=1)).t()                                                 # B x C (mean = centroid)
        exps = torch.exp(sim) * (nums > 0).float().t()
        prob = exps / (exps.sum(1, keepdim=True) + 1e-6)
        return F.nll_loss(torch.log(prob + 1e-6), self.labels[indexes])


class RandomMultipleGallerySampler(Sampler):
    """For every pseudo-identity: 1 anchor + (K-1) images preferably from other cameras."""
    def __init__(self, data, num_instances=4):
        self.K = num_instances
        self.pid_index, self.pid_cam = collections.defaultdict(list), collections.defaultdict(list)
        for i, (_, pid, cam) in enumerate(data):
            self.pid_index[pid].append(i); self.pid_cam[pid].append(cam)
        self.pids = list(self.pid_index)

    def __len__(self):
        return len(self.pids) * self.K

    def __iter__(self):
        ret = []
        for k in torch.randperm(len(self.pids)).tolist():
            idxs, cams = self.pid_index[self.pids[k]], self.pid_cam[self.pids[k]]
            j = random.randrange(len(idxs))
            ret.append(idxs[j])
            cand = [n for n, c in enumerate(cams) if c != cams[j]] or [n for n in range(len(idxs)) if n != j]
            if cand:
                ret.extend(idxs[n] for n in np.random.choice(cand, self.K - 1, replace=len(cand) < self.K - 1))
        return iter(ret)


EPS, EPS_GAP, K1, K2 = 0.6, 0.02, 30, 6


def compute_jaccard_distance(features, k1=K1, k2=K2):
    """k-reciprocal encoding (Zhong et al. CVPR'17) with sparse matrices -> dense N x N Jaccard distance."""
    N = features.size(0)
    f = F.normalize(features.float(), dim=1)
    rank = torch.cat([(f[s:s + 1024] @ f.t()).topk(k1 + 1, dim=1)[1].cpu() for s in range(0, N, 1024)]).numpy()
    fn = f.cpu().numpy()

    def k_reciprocal(i, k):
        fwd = rank[i, :k + 1]
        return fwd[np.where(rank[fwd, :k + 1] == i)[0]]

    nn_k1 = [k_reciprocal(i, k1) for i in range(N)]
    nn_half = [k_reciprocal(i, int(round(k1 / 2))) for i in range(N)]
    rows, cols, vals = [], [], []
    for i in range(N):
        idx, exp = nn_k1[i], [nn_k1[i]]
        for c in idx:
            cand = nn_half[c]
            if len(np.intersect1d(cand, idx)) > 2 / 3 * len(cand):
                exp.append(cand)
        exp = np.unique(np.concatenate(exp))
        w = np.exp(-(2 - 2 * fn[exp] @ fn[i]))
        rows.append(np.full(len(exp), i)); cols.append(exp); vals.append(w / w.sum())
    V = sp.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    if k2 > 1:                                                     # local query expansion
        A = sp.csr_matrix((np.full(N * k2, 1.0 / k2, np.float32), (np.repeat(np.arange(N), k2), rank[:, :k2].ravel())), shape=(N, N))
        V = (A @ V).tocsr()
    Vc = V.tocsc()
    jac = np.empty((N, N), np.float32)
    for i in range(N):
        tmp = np.zeros(N, np.float32)
        for k, v in zip(V.indices[V.indptr[i]:V.indptr[i + 1]], V.data[V.indptr[i]:V.indptr[i + 1]]):
            s, e = Vc.indptr[k], Vc.indptr[k + 1]
            tmp[Vc.indices[s:e]] += np.minimum(v, Vc.data[s:e])
        jac[i] = 1 - tmp / (2 - tmp)
    return np.clip(jac, 0, 1, out=jac)


def to_unique(labels):
    """DBSCAN outliers (-1) become their own single-instance 'clusters'."""
    labels = np.asarray(labels).copy()
    out = labels < 0
    labels[out] = labels.max() + 1 + np.arange(out.sum())
    return labels


def set_iou(a, b):
    """Per-sample IoU between the cluster containing i under labelling a and under labelling b."""
    _, ia, ca = np.unique(a, return_inverse=True, return_counts=True)
    _, ib, cb = np.unique(b, return_inverse=True, return_counts=True)
    _, ip, cp = np.unique(ia.astype(np.int64) * (ib.max() + 1) + ib, return_inverse=True, return_counts=True)
    inter = cp[ip]
    return inter / (ca[ia] + cb[ib] - inter)


def pseudo_label(features, indep_thres=None):
    t0 = time.time()
    jac = compute_jaccard_distance(features)
    run = lambda eps: to_unique(DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(jac))
    lab, tight, loose = run(EPS), run(EPS - EPS_GAP), run(EPS + EPS_GAP)
    del jac
    r_comp, r_indep = 1 - set_iou(lab, tight), 1 - set_iou(lab, loose)      # lower = more reliable
    C = lab.max() + 1
    clu_comp, clu_indep = np.full(C, np.inf), np.full(C, np.inf)
    np.minimum.at(clu_comp, lab, r_comp); np.minimum.at(clu_indep, lab, r_indep)
    if indep_thres is None:                                                  # fixed after the first epoch
        v = np.sort(clu_indep[np.bincount(lab, minlength=C) > 1])
        indep_thres = v[min(len(v) - 1, int(round(len(v) * 0.9)))] if len(v) else 1.0
    keep = (clu_indep[lab] <= indep_thres) & (r_comp <= clu_comp[lab])
    final = np.unique(to_unique(np.where(keep, lab, -1)), return_inverse=True)[1]
    sizes = np.bincount(final)
    stats = {'clusters': int((sizes > 1).sum()), 'unclustered': int((sizes == 1).sum()), 'cluster_time': time.time() - t0}
    print(f"   pseudo labels: {stats['clusters']} clusters, {stats['unclustered']} un-clustered ({stats['cluster_time']:.0f}s)")
    return final, indep_thres, stats


def run_spcl():
    # Adam 3.5e-4, x0.1 at epoch 20 and 40, 50 epochs x 400 iters, batch 64 (16 pseudo-ids x 4)
    EPOCHS, ITERS, EVAL_EVERY = (1, 20, 1) if QUICK else (50, 400, 10)
    name = tag('spcl')
    data = rc.build_data(ROOT, size=(256, 128), sampler='random')      # only query/gallery loaders are used
    unlabeled = [(p, -1, c) for p, _, c in data.train]                 # drop identity labels: keep path + camera
    train_tf, test_tf = rc.build_transforms((256, 128), train=True), rc.build_transforms((256, 128), train=False)

    model = SpCLNet().to(rc.DEVICE)
    memory = HybridMemory(2048, len(unlabeled)).to(rc.DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3.5e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    scaler = rc.make_scaler()
    history, best, indep_thres, start, eval_time = [], {'mAP': -1}, None, 0, 0.0
    ck = rc.load_ckpt(name)
    if ck:                                                  # resume: model + memory bank + optimiser state
        model.load_state_dict(ck['model']); memory.load_state_dict(ck['memory'])
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        history, best, indep_thres, start = ck['history'], ck['best'], ck['indep_thres'], ck['epoch']
    else:                                                   # initialise memory with ImageNet features
        init_loader = DataLoader(rc.ReIDDataset(unlabeled, test_tf), 256, shuffle=False, num_workers=4)
        memory.features = F.normalize(rc.extract_features(model, init_loader)[0], dim=1).to(rc.DEVICE)

    for ep in range(start, EPOCHS):
        labels, indep_thres, stats = pseudo_label(memory.features.clone(), indep_thres)
        memory.labels = torch.as_tensor(labels, dtype=torch.long, device=rc.DEVICE)
        pl_data = [(p, int(l), c) for (p, _, c), l in zip(unlabeled, labels)]
        loader = DataLoader(rc.ReIDDataset(pl_data, train_tf), 64, sampler=RandomMultipleGallerySampler(pl_data, 4),
                            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
        it, tot, t0 = iter(loader), 0.0, time.time()
        model.train()
        for _ in range(ITERS):
            try:
                imgs, _, _, idx = next(it)
            except StopIteration:
                it = iter(loader); imgs, _, _, idx = next(it)
            imgs, idx = imgs.to(rc.DEVICE, non_blocking=True), idx.to(rc.DEVICE)
            with rc.autocast():
                f = model(imgs)
            loss = memory(f, idx)                                   # memory loss in fp32 (outside autocast)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        rec = {'epoch': ep + 1, 'loss': tot / ITERS, 'lr': opt.param_groups[0]['lr'], 'time': time.time() - t0, **stats}
        sched.step()
        print(f"[{name}] epoch {ep + 1}/{EPOCHS}  loss {rec['loss']:.4f}  lr {rec['lr']:.1e}  {rec['time']:.0f}s")
        if (ep + 1) % EVAL_EVERY == 0 or ep + 1 == EPOCHS:
            t1 = time.time()
            rec.update(rc.test(model, data))
            eval_time = time.time() - t1
            if rec['mAP'] > best['mAP']:
                best = dict(rec); torch.save(model.state_dict(), f'{rc.OUT_DIR}/{name}_best.pth')
        history.append(rec)
        rc.save_ckpt(name, model=model.state_dict(), memory=memory.state_dict(), optimizer=opt.state_dict(),
                     scheduler=sched.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1,
                     history=history, best=best, indep_thres=indep_thres)
        json.dump({'history': history, 'best': best}, open(f'{rc.OUT_DIR}/{name}_history.json', 'w'), indent=1)
        del loader, it
        if ep + 1 < EPOCHS and rc.time_left() < 1.5 * (rec['time'] + stats['cluster_time']) + eval_time + 300:
            rc.STOPPED = True
            print(f'[{name}] TIME BUDGET REACHED after epoch {ep + 1}: checkpoint saved, stopping cleanly.')
            return

    rc.plot_history(history, name)
    plt.plot([h['epoch'] for h in history], [h['clusters'] for h in history], label='#clusters')
    plt.plot([h['epoch'] for h in history], [h['unclustered'] for h in history], label='#un-clustered')
    plt.axhline(751, ls='--', c='gray', label='true #IDs (751)'); plt.legend(); plt.xlabel('epoch')
    plt.title('SpCL pseudo-label statistics'); plt.savefig(f'{rc.OUT_DIR}/{name}_clusters.png', dpi=120); plt.show()
    rc.compare([('SpCL (unsupervised)', best, 88.1, 73.1)])


# ===== CELL 8 : RUN =====
RUNNERS = {'explore': run_explore, 'pcb': run_pcb, 'osnet': run_osnet, 'transreid': run_transreid, 'spcl': run_spcl}
todo = list(RUNNERS) if PAPER == 'all' else [PAPER]
if PAPER == 'all' and not QUICK:
    print('WARNING: full training of all papers takes ~15 h and will hit the 12 h Kaggle limit. Run one paper per commit.')
import traceback
for p in todo:
    print(f'\n{"=" * 30} {p.upper()} {"=" * 30}')
    t0 = time.time()
    try:
        RUNNERS[p]()
    except Exception:
        # Do not crash the notebook: a finished version keeps all checkpoints/results in its Output tab.
        err = traceback.format_exc()
        print(err)
        open(f'{rc.OUT_DIR}/ERROR_{p}.txt', 'w').write(err)
        print(f'!! {p} FAILED. Error saved to outputs/ERROR_{p}.txt (paste it in the chat). Checkpoints are kept.')
        if not QUICK:
            break
    gc.collect(); torch.cuda.empty_cache()
    print(f'{p} done in {(time.time() - t0) / 60:.1f} min')
    if rc.STOPPED:
        print('\n>> STOPPED BEFORE THE TIME LIMIT. To continue: open this notebook, "+ Add Input" -> Your Work -> '
              'Notebooks -> this notebook (latest version), keep the same PAPER, then Save & Run All again.')
        break
print('\nOutputs saved in', rc.OUT_DIR, ':', sorted(os.listdir(rc.OUT_DIR)))
