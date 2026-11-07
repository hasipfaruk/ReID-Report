# Run 1 results (interactive Kaggle session, 2x Tesla T4, JOB='all', QUICK=False)

Transcribed from the Cell 6 log screenshot (session stopped by Kaggle during transreid_prcc epoch 44+).
Format: mAP / R1 / R5 / R10 / mINP, all in %.

## TransReID on Market-1501 (120 epochs, 255.6 min, ~123 s/epoch, test 70 s)
| epoch | mAP | R1 | R5 | R10 | mINP |
|---|---|---|---|---|---|
| 20 | 84.77 | 93.38 | 97.57 | 98.55 | 60.75 |
| 40 | 87.25 | 94.18 | 97.95 | 98.81 | 65.52 |
| 60 | 87.51 | 94.27 | 98.04 | 98.99 | 66.04 |
| 80 | 88.56 | 94.86 | 98.22 | 98.93 | 67.79 |
| 100 | 88.84 | 94.89 | 98.25 | 98.90 | 68.54 |
| 120 (best) | 88.89 | 95.07 | 98.19 | 98.96 | 68.58 |
Paper (TransReID*, stride 12): R1 95.2, mAP 88.9.
Training loss: 10.72 (ep1) -> 0.10 (ep120).

## CAL on PRCC (60 epochs, 138.8 min, ~125 s/epoch, test ~60 s; CAL active from epoch 26)
| epoch | SC mAP | SC R1 | SC R5 | SC R10 | SC mINP | CC mAP | CC R1 | CC R5 | CC R10 | CC mINP |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 86.33 | 94.89 | 96.51 | 97.08 | 35.58 | 29.06 | 34.10 | 40.76 | 44.23 | 8.53 |
| 10 | 89.52 | 96.54 | 97.93 | 98.45 | 41.18 | 33.59 | 36.55 | 42.93 | 46.32 | 13.31 |
| 15 | 95.09 | 99.33 | 99.61 | 99.74 | 56.78 | 41.85 | 45.39 | 50.83 | 53.46 | 14.05 |
| 20 | 92.79 | 97.19 | 98.14 | 98.32 | 55.80 | 41.50 | 45.55 | 51.37 | 55.04 | 13.07 |
| 25 | 96.47 | 99.35 | 99.64 | 99.72 | 64.06 | 42.81 | 45.98 | 51.03 | 53.60 | 13.25 |
| 30 | 99.43 | 100.00 | 100.00 | 100.00 | 89.39 | 51.47 | 50.55 | 55.77 | 57.95 | 34.32 |
| 35 | 99.48 | 100.00 | 100.00 | 100.00 | 89.39 | 54.36 | 53.32 | 58.54 | 61.11 | 36.30 |
| 40 | 99.63 | 100.00 | 100.00 | 100.00 | 92.03 | 54.00 | 53.82 | 57.32 | 58.62 | 36.84 |
| 45 | 99.66 | 100.00 | 100.00 | 100.00 | 92.19 | 54.19 | 52.72 | 56.51 | 58.23 | 36.74 |
| 50 | 99.66 | 100.00 | 100.00 | 100.00 | 92.28 | 53.57 | 52.19 | 57.86 | 60.32 | 35.93 |
| 55 (best) | 99.73 | 100.00 | 100.00 | 100.00 | 93.10 | 54.93 | 53.26 | 57.66 | 59.75 | 38.18 |
| 60 | 99.75 | 100.00 | 100.00 | 100.00 | 92.92 | 54.23 | 52.81 | 56.59 | 58.85 | 36.98 |
Paper: SC R1 100.0 / mAP 99.8; CC R1 55.2 / mAP 55.8.
Loss: 3.31 (ep1) -> 0.90 (ep25), jumps to 5.38 at ep26 when the clothes classifier + adversarial loss switch on, -> 1.08 (ep60).

## Baseline (same network, no CAL) on PRCC (60 epochs, 137.4 min)
| epoch | SC mAP | SC R1 | SC R5 | SC R10 | SC mINP | CC mAP | CC R1 | CC R5 | CC R10 | CC mINP |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 86.65 | 96.15 | 97.42 | 97.96 | 33.35 | 32.25 | 38.92 | 46.43 | 49.51 | 10.23 |
| 10 | 94.03 | 98.06 | 99.28 | 99.48 | 53.52 | 36.55 | 39.60 | 43.66 | 45.81 | 15.13 |
| 15 | 94.33 | 98.58 | 99.15 | 99.38 | 54.22 | 33.15 | 34.04 | 38.27 | 40.84 | 13.02 |
| 20 | 94.37 | 98.37 | 99.12 | 99.38 | 53.00 | 33.30 | 35.85 | 43.16 | 46.18 | 9.16 |
| 25 | 96.77 | 99.74 | 99.87 | 99.90 | 63.12 | 40.26 | 42.79 | 46.99 | 49.28 | 13.86 |
| 30 | 96.98 | 99.64 | 99.82 | 99.87 | 66.95 | 42.50 | 44.96 | 50.30 | 52.81 | 16.56 |
| 35 | 97.78 | 99.69 | 99.87 | 99.92 | 71.88 | 43.44 | 44.20 | 48.94 | 52.07 | 17.57 |
| 40 | 97.57 | 99.51 | 99.79 | 99.87 | 72.69 | 43.26 | 43.66 | 48.86 | 52.02 | 18.84 |
| 45 (best) | 97.97 | 99.77 | 99.92 | 99.95 | 75.24 | 45.75 | 45.70 | 50.64 | 54.05 | 22.26 |
| 50 | 98.01 | 99.82 | 99.90 | 99.92 | 74.60 | 45.43 | 46.54 | 51.03 | 53.66 | 20.80 |
| 55 | 97.93 | 99.69 | 99.87 | 99.92 | 75.26 | 45.68 | 46.88 | 51.51 | 54.11 | 22.29 |
| 60 | 97.97 | 99.74 | 99.90 | 99.95 | 75.12 | 45.56 | 46.57 | 51.65 | 54.22 | 19.60 |
Loss: 3.21 (ep1) -> 0.86 (ep60).
CAL vs baseline (best, clothes-changing): R1 +7.56, mAP +9.18, mINP +15.92.

## TransReID on PRCC (INCOMPLETE: stopped after epoch 44 of 120, ~180 s/epoch)
| epoch | SC mAP | SC R1 | SC R5 | SC R10 | SC mINP | CC mAP | CC R1 | CC R5 | CC R10 | CC mINP |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 98.73 | 99.85 | 100.00 | 100.00 | 86.80 | 47.57 | 46.71 | 50.86 | 52.67 | 32.04 |
| 40 | 99.17 | 99.97 | 100.00 | 100.00 | 89.57 | 47.77 | 45.98 | 50.64 | 53.15 | 32.53 |
Loss: 8.98 (ep1) -> 0.02 (ep44): fits the 150 training identities almost perfectly.

## Dataset statistics (explore job)
Market-1501: train 12,936 imgs / 751 ids / 6 cams; query 3,368 / 750; gallery 15,913 (13,115 + 2,798 distractors) / 750.
PRCC: train 17,896 / 150 ids / 3 cams / 300 outfits (2 per id, 119.3 imgs per id); test 71 ids: gallery A 3,384, same-clothes query B 3,873, clothes-changing query C 3,543.

## TransReID on PRCC, COMPLETE (Kaggle commit version, 120 epochs, 349.3 min, ~170 s/epoch)
| epoch | SC mAP | SC R1 | SC R5 | SC R10 | SC mINP | CC mAP | CC R1 | CC R5 | CC R10 | CC mINP |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 99.10 | 100.00 | 100.00 | 100.00 | 89.91 | 49.84 | 48.66 | 54.39 | 56.62 | 32.73 |
| 40 | 99.29 | 99.97 | 100.00 | 100.00 | 90.85 | 48.67 | 46.63 | 53.82 | 56.48 | 33.71 |
| 60 | 99.12 | 99.95 | 100.00 | 100.00 | 90.22 | 50.11 | 49.03 | 54.78 | 58.26 | 33.36 |
| 80 | 99.19 | 99.95 | 100.00 | 100.00 | 90.61 | 50.12 | 48.91 | 54.76 | 57.92 | 34.20 |
| 100 | 99.11 | 99.97 | 100.00 | 100.00 | 90.35 | 50.24 | 48.74 | 54.25 | 56.70 | 33.79 |
| 120 (best by CC mAP) | 99.12 | 99.97 | 100.00 | 100.00 | 90.33 | 50.41 | 48.60 | 54.61 | 57.24 | 34.04 |
Loss 9.01 (ep1) -> 0.011 (ep120). Files saved in that version: transreid_prcc_best.pth, _ckpt.pth, _curves.png, _history.json.
(The interrupted interactive attempt reached CC mAP 47.57/47.77 at ep20/40; this full run supersedes it.)

## Final comparison on PRCC clothes-changing (best checkpoints)
| method | R1 | mAP | mINP |
|---|---|---|---|
| Baseline ResNet-50 (no CAL) | 45.70 | 45.75 | 22.26 |
| TransReID (ViT-B/16) | 48.60 | 50.41 | 34.04 |
| CAL (ResNet-50 + CAL) | 53.26 | 54.93 | 38.18 |
| CAL paper | 55.2 | 55.8 | - |

## Paper numbers VERIFIED from the PDFs (2026-10-05)
- TransReID (arXiv 2102.04378, Table "comparison with state of the art", ViT-B/16 block): Market-1501 TransReID 256x128 mAP 88.2 / R1 95.0; TransReID* (stride 12) 256x128 mAP 88.9 / R1 95.2; 384x128 88.8/95.0 and 89.5/95.2. (DeiT-B/16 block: TransReID* 256x128 88.4/95.0.)
- CAL (arXiv 2204.06890), Table 2/3 PRCC: CAL SC 100 / 99.8, CC 55.2 / 55.8; LTCC general 74.2 / 40.8, CC 40.1 / 18.0.
- CAL Table 3 ablation PRCC (top-1 / mAP): baseline SC 99.8/97.9, CC 45.6/43.3; w/ clothes classifier CC 33.1/37.4; CAL(-LC) CC 19.3/13.1; Triplet Loss CC 48.6/49.7.
- CAL Table 4: Market-1501 baseline 92.2/78.7, CAL 94.5/87.3.
