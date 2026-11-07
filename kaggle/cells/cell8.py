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
