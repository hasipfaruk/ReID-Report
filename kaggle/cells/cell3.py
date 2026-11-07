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
