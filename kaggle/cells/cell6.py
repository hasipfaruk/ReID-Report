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
