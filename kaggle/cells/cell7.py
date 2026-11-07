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
