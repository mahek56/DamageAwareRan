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
    sample_ids = np.asarray(sample_ids)
    checkpoints = np.asarray(checkpoints)
    scores = np.asarray(scores, dtype=np.float32)
    if sample_ids.ndim != 1:
        raise ValueError('sample_ids must be 1D')
    if checkpoints.ndim != 1:
        raise ValueError('checkpoints must be 1D')
    if scores.ndim != 2:
        raise ValueError('scores must have shape (n_samples, n_checkpoints)')
    if scores.shape[0] != len(sample_ids):
        raise ValueError(f'scores rows ({scores.shape[0]}) must match sample_ids ({len(sample_ids)})')
    if scores.shape[1] != len(checkpoints):
        raise ValueError(f'scores columns ({scores.shape[1]}) must match checkpoints ({len(checkpoints)})')
    if final_probs is not None:
        final_probs = np.asarray(final_probs, dtype=np.float32)
        if final_probs.ndim != 2:
            raise ValueError('final_probs must have shape (n_samples, n_labels)')
        if final_probs.shape[0] != len(sample_ids):
            raise ValueError(f'final_probs rows ({final_probs.shape[0]}) must match sample_ids ({len(sample_ids)})')
        if final_probs.shape[1] != len(LABELS):
            raise ValueError(f'final_probs columns ({final_probs.shape[1]}) must match LABELS ({len(LABELS)})')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(sample_ids=sample_ids, checkpoints=checkpoints, scores=scores)
    if final_probs is not None:
        kw['final_probs'] = final_probs
    np.savez(path, **kw)
def load_scores(path):
    z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}
