"""Report figures drawn from the transcribed Kaggle logs (results/run1_interactive_log_results.md). Black and grey only."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
                     'xtick.color': 'black', 'ytick.color': 'black', 'savefig.dpi': 200})

# ---------------------------------------------------------------- data (from the Kaggle logs)
tr_market = {'ep': [20, 40, 60, 80, 100, 120],
             'mAP': [84.77, 87.25, 87.51, 88.56, 88.84, 88.89], 'R1': [93.38, 94.18, 94.27, 94.86, 94.89, 95.07]}
ep5 = list(range(5, 65, 5))
cal_cc_map = [29.06, 33.59, 41.85, 41.50, 42.81, 51.47, 54.36, 54.00, 54.19, 53.57, 54.93, 54.23]
cal_cc_r1 = [34.10, 36.55, 45.39, 45.55, 45.98, 50.55, 53.32, 53.82, 52.72, 52.19, 53.26, 52.81]
base_cc_map = [32.25, 36.55, 33.15, 33.30, 40.26, 42.50, 43.44, 43.26, 45.75, 45.43, 45.68, 45.56]
base_cc_r1 = [38.92, 39.60, 34.04, 35.85, 42.79, 44.96, 44.20, 43.66, 45.70, 46.54, 46.88, 46.57]
tr_prcc = {'ep': [20, 40, 60, 80, 100, 120], 'mAP': [49.84, 48.67, 50.11, 50.12, 50.24, 50.41],
           'R1': [48.66, 46.63, 49.03, 48.91, 48.74, 48.60]}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------- Fig: TransReID on Market-1501
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.plot(tr_market['ep'], tr_market['R1'], 's--', color='black', label='Rank-1 (ours)')
ax.plot(tr_market['ep'], tr_market['mAP'], 'o-', color='black', label='mAP (ours)')
ax.axhline(95.2, color='0.5', ls=':', lw=1.2); ax.text(21, 95.35, 'paper Rank-1 95.2', fontsize=8, color='0.3')
ax.axhline(88.9, color='0.5', ls=':', lw=1.2); ax.text(21, 89.05, 'paper mAP 88.9', fontsize=8, color='0.3')
ax.set(xlabel='epoch', ylabel='%', title='TransReID on Market-1501: test accuracy during training', ylim=(83, 97))
ax.legend(loc='lower right', fontsize=8, frameon=False)
save(fig, 'fig_transreid_market_curve.png')

# ---------------------------------------------------------------- Fig: PRCC clothes-changing mAP over training
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(ep5, cal_cc_map, 'o-', color='black', label='CAL')
ax.plot(ep5, base_cc_map, 's--', color='0.45', label='Baseline (no CAL)')
ax.plot(tr_prcc['ep'], tr_prcc['mAP'], '^:', color='black', label='TransReID (120 epochs)')
ax.axvline(25, color='0.6', lw=1); ax.text(25.8, 30, 'CAL switched on\n(epoch 26)', fontsize=8, color='0.3')
ax.axhline(55.8, color='0.6', ls=':', lw=1); ax.text(80, 56.3, 'CAL paper mAP 55.8', fontsize=8, color='0.3')
ax.set(xlabel='epoch', ylabel='mAP (%)', title='PRCC clothes-changing setting: mAP during training', ylim=(26, 60))
ax.legend(loc='lower right', fontsize=8, frameon=False)
save(fig, 'fig_prcc_cc_curves.png')

# ---------------------------------------------------------------- Fig: PRCC comparison bars
methods = ['Baseline\n(ResNet-50)', 'TransReID\n(ViT-B/16)', 'CAL\n(ours)', 'CAL\n(paper)']
sc = {'R1': [99.77, 99.97, 100.0, 100.0], 'mAP': [97.97, 99.12, 99.73, 99.8]}
cc = {'R1': [45.70, 48.60, 53.26, 55.2], 'mAP': [45.75, 50.41, 54.93, 55.8]}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, (title, d) in zip(axes, (('same clothes', sc), ('clothes-changing', cc))):
    x = range(len(methods))
    b1 = ax.bar([i - 0.19 for i in x], d['R1'], 0.38, color='0.35', edgecolor='black', label='Rank-1')
    b2 = ax.bar([i + 0.19 for i in x], d['mAP'], 0.38, color='white', edgecolor='black', hatch='//', label='mAP')
    ax.bar_label(b1, fmt='%.1f', fontsize=7, padding=1); ax.bar_label(b2, fmt='%.1f', fontsize=7, padding=1)
    ax.set_xticks(list(x), methods, fontsize=8); ax.set_title(f'PRCC, {title}'); ax.set_ylim(0, 112)
axes[0].set_ylabel('%'); axes[1].legend(fontsize=8, frameon=False, loc='upper left')
save(fig, 'fig_prcc_comparison.png')


# ---------------------------------------------------------------- method diagrams
def box(ax, x, y, w, h, text, fill='white', fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=fill, ec='black', lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal')


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='black', lw=1))


# TransReID
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(-1, 103); ax.set_ylim(0, 50); ax.axis('off')
box(ax, 0, 20, 11, 10, 'image\n256x128')
box(ax, 14, 20, 15, 10, 'overlapping\n16x16 patches\nstride 12\n(21x11 = 231)')
box(ax, 32, 27, 15, 9, '+ position emb.\n+ SIE camera emb.\n(x 3.0)', fill='0.9')
box(ax, 32, 14, 15, 9, '[cls] token +\n231 patch tokens')
box(ax, 50, 20, 13, 10, 'ViT-B/16\nblocks 1 to 11')
box(ax, 67, 34, 14, 9, 'block 12 (copy 1)\nglobal branch')
box(ax, 66, 5, 15, 16, 'JPM: shift 5 +\npatch shuffle,\n4 groups, block 12\n(copy 2)')
box(ax, 85, 36, 15, 6, 'global feature\nBNNeck + ID + triplet', fs=7)
box(ax, 85, 9, 15, 8, '4 local features\nBNNeck + ID + triplet', fs=7)
arrow(ax, 11, 25, 14, 25); arrow(ax, 29, 25, 32, 28); arrow(ax, 29, 25, 32, 19)
arrow(ax, 47, 31, 50, 26); arrow(ax, 47, 18, 50, 24)
arrow(ax, 63, 27, 67, 38); arrow(ax, 63, 23, 67, 14)
arrow(ax, 81, 38.5, 84, 39); arrow(ax, 81, 13, 84, 13)
ax.text(50, 47, 'TransReID: test feature = [global, local/4 x 4] (3840-d)', ha='center', fontsize=9)
save(fig, 'fig_transreid_architecture.png')

# CAL
fig, ax = plt.subplots(figsize=(7.4, 3.8)); ax.set_xlim(-1, 103); ax.set_ylim(0, 52); ax.axis('off')
box(ax, 0, 22, 11, 10, 'image\n384x192')
box(ax, 14, 22, 15, 10, 'ResNet-50\n(last stride 1)')
box(ax, 32, 22, 15, 10, 'max + avg pool\nconcat 4096-d\n+ BN')
box(ax, 52, 36, 19, 9, 'identity classifier\n(linear)')
box(ax, 52, 6, 19, 12, 'clothes classifier\n(cosine, one class\nper outfit)', fill='0.9')
box(ax, 76, 36, 24, 9, 'L_ID: cross-entropy\n+ label smoothing', fs=7)
box(ax, 76, 20, 24, 10, 'L_C: classify outfit\n(features detached,\nupdates classifier only)', fs=7)
box(ax, 76, 3, 24, 12, 'L_CA (adversarial):\npull feature to ALL outfits\nof same person, push from\noutfits of other people', fs=7)
arrow(ax, 11, 27, 14, 27); arrow(ax, 29, 27, 32, 27)
arrow(ax, 47, 29, 52, 40); arrow(ax, 47, 25, 52, 13)
arrow(ax, 71, 40.5, 76, 40.5); arrow(ax, 71, 14, 76, 24); arrow(ax, 71, 11, 76, 9)
ax.text(50, 49.5, 'CAL training: epochs 1 to 25 use L_ID only; from epoch 26: L_ID + L_C + L_CA', ha='center', fontsize=9)
save(fig, 'fig_cal_architecture.png')
print('figures written to', OUT, sorted(os.listdir(OUT)))


# ================================================================ FAIR COMPARISON (read from the Kaggle result files)
import json
RUN = os.path.join(os.path.dirname(os.path.dirname(OUT)), 'results', 'fair_run_s0', 'outputs')
fair = {m: json.load(open(os.path.join(RUN, f'fair_{m}_s0_history.json')))['history'] for m in ('baseline', 'transreid', 'cal')}
names = {'baseline': 'Baseline (ResNet-50)', 'transreid': 'TransReID (ViT-B/16)', 'cal': 'CAL (ResNet-50)'}
styles = {'baseline': ('s--', '0.45'), 'transreid': ('^:', 'black'), 'cal': ('o-', 'black')}
ev = {m: [r for r in h if 'mAP' in r] for m, h in fair.items()}

# Fig: clothes-changing mAP and Rank-1 during training, fair protocol
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)
for ax, key, lab in zip(axes, ('clothes_changing/mAP', 'clothes_changing/R1'), ('mAP (%)', 'Rank-1 (%)')):
    for m in ('baseline', 'transreid', 'cal'):
        st, c = styles[m]
        ax.plot([r['epoch'] for r in ev[m]], [r[key] * 100 for r in ev[m]], st, color=c, label=names[m], ms=5)
    ax.axvline(25.5, color='0.65', lw=1)
    ax.set(xlabel='epoch', ylabel=lab, ylim=(37, 54), xticks=range(10, 70, 10))
axes[0].text(26.5, 38, 'CAL loss\nswitched on', fontsize=7, color='0.3')
axes[0].set_title('Clothes-changing mAP', fontsize=10); axes[1].set_title('Clothes-changing Rank-1', fontsize=10)
axes[1].legend(fontsize=7, frameon=False, loc='lower right')
save(fig, 'fig_fair_curves.png')

# Fig: final results (epoch 60) under the fair protocol
last = {m: ev[m][-1] for m in ev}
order = ['baseline', 'transreid', 'cal']
fig, ax = plt.subplots(figsize=(6.6, 3.2))
for i, (k, lab, fc, hatch) in enumerate((('R1', 'Rank-1', '0.35', ''), ('mAP', 'mAP', 'white', '//'), ('mINP', 'mINP', '0.8', ''))):
    vals = [last[m][f'clothes_changing/{k}'] * 100 for m in order]
    b = ax.bar([j + (i - 1) * 0.26 for j in range(3)], vals, 0.26, color=fc, edgecolor='black', hatch=hatch, label=lab)
    ax.bar_label(b, fmt='%.1f', fontsize=7, padding=1)
ax.set_xticks(range(3), [names[m].replace(' (', '\n(') for m in order], fontsize=8)
ax.set(ylabel='%', ylim=(0, 62), title='PRCC clothes-changing, fair protocol, epoch 60')
ax.legend(fontsize=8, frameon=False, ncol=3, loc='upper right')
save(fig, 'fig_fair_comparison.png')

# Fig: same model, original settings vs fair protocol (clothes-changing mAP at the last epoch)
orig_last = {'baseline': 45.56, 'transreid': 50.11, 'cal': 54.23}      # epoch 60 of the original-settings runs
fig, ax = plt.subplots(figsize=(6.2, 3.0))
b1 = ax.bar([j - 0.19 for j in range(3)], [orig_last[m] for m in order], 0.38, color='0.35', edgecolor='black', label="each paper's own settings")
b2 = ax.bar([j + 0.19 for j in range(3)], [last[m]['clothes_changing/mAP'] * 100 for m in order], 0.38, color='white',
            edgecolor='black', hatch='//', label='fair protocol')
ax.bar_label(b1, fmt='%.1f', fontsize=7, padding=1); ax.bar_label(b2, fmt='%.1f', fontsize=7, padding=1)
ax.set_xticks(range(3), [names[m].replace(' (', '\n(') for m in order], fontsize=8)
ax.set(ylabel='clothes-changing mAP (%)', ylim=(0, 72), title='Effect of the training protocol (epoch 60)')
ax.legend(fontsize=8, frameon=False, loc='upper center', ncol=2)
save(fig, 'fig_protocol_effect.png')
print('fair figures written')
