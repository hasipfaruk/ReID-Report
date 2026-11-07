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
