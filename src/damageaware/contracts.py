"""Data contracts shared by all roles. Do not edit without a CHANGE REQUEST."""
import json
from pathlib import Path
import numpy as np, pandas as pd
LABELS = ['benign', 'ransomware', 'rat', 'spyware']
N_CAP, PAD, UNK = 1024, 0, 1
def default_checkpoints(n_cap=N_CAP):
    return list(range(8, 129, 8)) + list(range(160, n_cap + 1, 32))
def load_meta(d): return pd.read_csv(Path(d) / 'meta.csv')
def load_trace(d, sid):
    z = np.load(Path(d) / 'traces' / f'{sid}.npz')
    return {'tok': z['tok'].astype(np.int32), 'dmg': z['dmg'].astype(np.int32)}
def load_json(d, name):
    with open(Path(d) / name) as f: return json.load(f)
def save_scores(path, sample_ids, checkpoints, scores, final_probs=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(sample_ids=np.array(sample_ids), checkpoints=np.array(checkpoints), scores=np.asarray(scores, np.float32))
    if final_probs is not None: kw['final_probs'] = np.asarray(final_probs, np.float32)
    np.savez(path, **kw)
def load_scores(path):
    z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}
