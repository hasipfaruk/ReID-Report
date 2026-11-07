# Person ReID: TransReID (ICCV 2021) and CAL (CVPR 2022) on Kaggle

| # | Paper | Venue | Problem | Datasets used here | Paper result (R1 / mAP) |
|---|-------|-------|---------|--------------------|--------------------------|
| 1 | TransReID: Transformer-based Object Re-Identification (He et al.) | ICCV 2021 | standard person ReID | Market-1501 | 95.2 / 88.9 |
| 2 | Clothes-Changing Person Re-identification with RGB Modality Only, "CAL" (Gu et al.) | CVPR 2022 | clothes-changing person ReID | PRCC (LTCC optional) | PRCC clothes-changing 55.2 / 55.8, same-clothes 100.0 / 99.8 |

All code is in **`kaggle/reid_kaggle_notebook.py`**. It has 6 cells, separated by `# ===== CELL n =====` markers.

| Cell | Content | Edit it? |
|------|---------|----------|
| 1 | Shared code: datasets, losses, evaluation, training, checkpoints. It starts with `%%writefile`. | no |
| 2 | **Switch**: `JOB`, `QUICK`, `MAX_HOURS` | **yes, only this cell** |
| 3 | Dataset exploration | no |
| 4 | Paper 1: TransReID | no |
| 5 | Paper 2: CAL | no |
| 6 | Run: executes the job chosen in Cell 2 | no |

## 1. Datasets (all free on Kaggle, nothing to download to your PC)
| Dataset | Needed for | Kaggle link (attach with "+ Add Input") |
|---------|-----------|------------------------------------------|
| Market-1501 | TransReID | https://www.kaggle.com/datasets/iiierie/market-1501-v15-09-15 (backup: rayiooo/reid_market-1501) |
| PRCC | CAL | https://www.kaggle.com/datasets/lyqassf/prccdata1 (complete: rgb train 17,896, val 5,002, test A/B/C) |
| LTCC (optional) | CAL | Not on Kaggle. Sign the release agreement at https://naiq.github.io/LTCC_Perosn_ReID.html and email the authors. When you receive it, upload it as a private Kaggle dataset (+ Create, New Dataset) and attach it. |

The code finds every attached dataset automatically. Cell 2 prints which ones it found.

## 2. One-time setup
1. Kaggle account with **phone verification** (needed for GPU and internet).
2. Create a new notebook.
3. Right-hand panel settings:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
   - **Persistence: Files only**
4. **+ Add Input**: attach **Market-1501** and **PRCC** (links above).
5. Create 6 code cells and paste each block. Cell 1 must start with `%%writefile /kaggle/working/reid_common.py` on its very first line.

## 3. Step A: quick test (about 15 to 20 min)
Set `JOB = 'all'` and `QUICK = True` in Cell 2, then click **Run All**. Every job runs 2 short epochs (30 iterations each) plus a full test. If something fails, paste the error, or the file `outputs/ERROR_<job>.txt`, into the chat.

## 4. Step B: real training (one job per "Save Version, Save & Run All")
Set `QUICK = False` and pick a `JOB`:

| Order | JOB | What it gives the report | Approx. time on T4 |
|-------|-----|--------------------------|--------------------|
| 1 | `'explore'` | dataset statistics + sample images | 3 min |
| 2 | `'transreid_market'` | Paper 1 reproduction | 4 to 6 h |
| 3 | `'cal_prcc'` | Paper 2 reproduction | 3 to 4 h |
| 4 | `'baseline_prcc'` | ablation: CAL vs same network without CAL | 3 to 4 h |
| 5 | `'transreid_prcc'` | how TransReID behaves when people change clothes | 6 to 9 h (may need 2 runs, resume is automatic) |

The total is about 20 to 25 GPU hours, within the 30 h weekly Kaggle quota. If needed, spread it over two weeks.

## 5. If a run stops (time limit, GPU shutdown, error): nothing is lost
- A full checkpoint is saved after every epoch (`outputs/<job>_ckpt.pth`).
- Training stops by itself before the 12 h limit (`MAX_HOURS`), so the version finishes and keeps its outputs.
- Errors are caught and written to `outputs/ERROR_<job>.txt`, and checkpoints are kept.

**To continue:**
1. Open the notebook.
2. **+ Add Input, Your Work, Notebooks**: add this notebook's latest version. Remove older attached versions.
3. Keep the same `JOB` with `QUICK = False`.
4. **Save & Run All**. It prints `RESUMING <job> from epoch N`.

## 6. Report run (after all training jobs are finished)
Cells 7 and 8 create the report material from the trained models.
1. **+ Add Input, Your Work, Notebooks**: attach the saved versions of this notebook that contain the finished jobs. The latest version is enough if each run resumed from the previous one.
2. In Cell 2, set `JOB = 'explore'` and `QUICK = False`.
3. Click **Save & Run All** (about 20 min). Cell 2 copies every earlier `*_best.pth` and `*_history.json` into `outputs/`. Then:
   - **Cell 7** writes `results_summary.csv` (all metrics next to the paper numbers), `results_<dataset>.png` (bar charts), and `samples_*.png` (example persons and the PRCC protocols).
   - **Cell 8** writes `retrieval_<job>.png`: 5 queries with their top-10 matches. A thick solid frame with a tick marks the correct person; a dashed frame with a cross marks a wrong person. All methods use the same queries.

## 7. Results for the report
In the saved version, open the **Output** tab and download `outputs/`:
- `<job>_history.json`: loss per epoch and all test metrics
- `<job>_curves.png`: training curves, black and grey only
- `<job>_best.pth`: trained weights
- `dataset_stats.json`, `<dataset>_stats.png`: dataset statistics

## Evaluation protocols
- **Market-1501**: single query; gallery images with the same id and same camera are ignored. No re-ranking, no flip (as in TransReID).
- **PRCC**: gallery = camera A. *Same-clothes* query = camera B; *clothes-changing* query = camera C. The full gallery is used, as in the CAL code.
- **LTCC**: *general* (standard protocol) and *clothes-changing* (also ignores same-id images with the same clothes).
- CAL uses flip at test time (as in its official code). Metrics: Rank-1/5/10, mAP, mINP.

## 8. FAIR COMPARISON on PRCC (supervisor's request)
Task: clothes-changing ReID on PRCC. Models: ResNet-50 baseline, TransReID, CAL, all retrained with the
**same protocol** (the `FAIR` dict in Cell 2):

| Same for all three | Kept per method (part of the method) |
|---|---|
| PRCC train set, input 256x128, batch 64 = 16 ids x 4 imgs | backbone (ResNet-50 / ViT-B/16) |
| flip + pad 10/random crop + random erasing 0.5 | losses (CE / CE+triplet+JPM / CE+CAL) |
| 60 epochs, evaluation every 10 epochs | optimiser + lr schedule (Adam step / SGD cosine) |
| flip at test, cosine distance, full gallery | input normalisation of the pre-trained weights |
| the **last epoch** is reported (no model selection on the test set) | |

Steps:
1. Cell 2: `JOB = 'fair_all'`, `QUICK = True`, Run All (about 10 min smoke test).
2. Cell 2: `JOB = 'fair_all'`, `QUICK = False`, `SEED = 0`, Save Version, Save & Run All (about 6 to 7 h).
3. Optional, for mean +- std: repeat step 2 with `SEED = 1` and `SEED = 2`. Attach the earlier version
   first (+ Add Input, Your Work), so its files are copied in.
4. Cell 7 writes `fair_comparison_prcc.csv`, `fair_comparison_prcc.png` and `fair_curves_prcc.png`.
   Cell 8 writes `retrieval_fair_<method>_s<seed>.png`. Cell 9 zips everything.
