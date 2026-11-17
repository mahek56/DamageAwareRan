"""Synthetic data in the real contract format, so all roles can start in parallel.
Usage: PYTHONPATH=src python scripts/make_stub_data.py [--out DIR] [--scores]"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from damageaware.contracts import LABELS, N_CAP, default_checkpoints, load_meta, load_trace, save_scores
def make(out, n=600, seed=0):
    rng = np.random.default_rng(seed); out = Path(out); (out / 'traces').mkdir(parents=True, exist_ok=True); V = 60
    labels = rng.choice(LABELS, size=n, p=[.3, .25, .25, .2]); clusters = rng.integers(0, n // 5, size=n)
    cl = np.unique(clusters); rng.shuffle(cl); c1, c2 = int(.7 * len(cl)), int(.85 * len(cl))
    cs = {c: 'train' if i < c1 else 'val' if i < c2 else 'test' for i, c in enumerate(cl)}; rows = []
    for i in range(n):
        sid = f's{i:04d}'; L = int(rng.integers(200, N_CAP + 1)); tok = rng.integers(4, V, size=L).astype(np.int32); dmg = np.zeros(L, np.int32)
        if labels[i] == 'ransomware':
            s = int(rng.integers(30, max(31, L - 100))); tok[s:s + 80:2] = 3; dmg[s:s + 80:2] = 1
        else:
            k = rng.integers(0, L, size=int(rng.integers(0, 6))); tok[k] = 3; dmg[k] = 1
        np.savez(out / 'traces' / f'{sid}.npz', tok=tok, dmg=dmg)
        rows.append(dict(sample_id=sid, label=labels[i], split=cs[clusters[i]], split_random=str(rng.choice(['train', 'val', 'test'], p=[.7, .15, .15])),
                         cluster_id=int(clusters[i]), n_events=L, total_damage_full=int(dmg.sum())))
    pd.DataFrame(rows).to_csv(out / 'meta.csv', index=False)
    json.dump({'<pad>': 0, '<unk>': 1, 'API_write': 3, **{f'API_{j}': j for j in range(4, V)}}, open(out / 'vocab.json', 'w'))
    json.dump(default_checkpoints(), open(out / 'checkpoints.json', 'w'))
def make_scores(d, seed=0):
    """Fake score files for two pretend models so R4 can build the evaluator. A reacts faster than B."""
    rng = np.random.default_rng(seed); meta = load_meta(d); cps = default_checkpoints()
    for name, k in (('stubA', 8.0), ('stubB', 16.0)):
        for split in ('val', 'test'):
            ids = meta[meta.split == split].sample_id.tolist(); S = []
            for sid in ids:
                dmg = load_trace(d, sid)['dmg']; cum = np.cumsum(dmg)
                S.append([min(1, cum[min(c, len(dmg)) - 1] / k + rng.normal(0, .03)) for c in cps])
            save_scores(f'results/scores/{name}_{split}_seed0.npz', ids, cps, np.clip(S, 0, 1))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='data/processed_stub'); ap.add_argument('--scores', action='store_true'); a = ap.parse_args()
    make(a.out)
    if a.scores: make_scores(a.out)
    print('stub data written to', a.out)
