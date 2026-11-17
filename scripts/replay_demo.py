"""Plot per-model prefix scores and cumulative damage for a single trace."""

import argparse
from importlib import import_module
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _default_sample_id(meta):
    candidates = meta[(meta['label'] == 'ransomware') & (meta['split'] == 'test')]
    if candidates.empty:
        raise ValueError('No ransomware traces are available in the test split.')
    return candidates.iloc[0]['sample_id']


def _load_score_payload(score_path):
    if not Path(score_path).exists():
        raise FileNotFoundError(f'Missing score file: {score_path}')
    with np.load(score_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    return payload


def _compute_tau(model_name, scores_dir, data_dir, seed, checkpoints, meta, alpha=0.05):
    try:
        thresholds_module = import_module('damageaware.eval.thresholds')
    except ModuleNotFoundError as error:
        if error.name != 'damageaware.eval.thresholds':
            raise
        return None, 'Tau unavailable: src/damageaware/eval/thresholds.py is missing; plotting without tau.'

    choose_tau = getattr(thresholds_module, 'choose_tau', None)
    if choose_tau is None:
        return None, 'Tau unavailable: choose_tau is missing from src/damageaware/eval/thresholds.py; plotting without tau.'

    val_benign = meta[(meta['split'] == 'val') & (meta['label'] == 'benign')]
    if val_benign.empty:
        return None, 'Tau unavailable: no benign validation traces are available; plotting without tau.'

    score_path = Path(scores_dir) / f'{model_name}_val_seed{seed}.npz'
    payload = _load_score_payload(score_path)
    sample_ids = np.asarray(payload['sample_ids'])
    score_matrix = np.asarray(payload['scores'], dtype=np.float32)
    if 'checkpoints' in payload:
        file_checkpoints = np.asarray(payload['checkpoints'])
        if not np.array_equal(file_checkpoints, checkpoints):
            raise ValueError(
                f'{model_name} validation checkpoint mismatch: {file_checkpoints.tolist()} != {checkpoints}'
            )

    benign_scores = []
    for sample_id in val_benign['sample_id']:
        if sample_id not in sample_ids.tolist():
            continue
        row_index = np.where(sample_ids == sample_id)[0][0]
        benign_scores.append(score_matrix[row_index])
    if not benign_scores:
        return None, 'Tau unavailable: no benign validation scores matched the score files; plotting without tau.'

    val_scores = np.asarray(benign_scores, dtype=np.float32)
    candidates = []
    if 'alpha' in inspect.signature(choose_tau).parameters:
        candidates.append({'scores': val_scores, 'alpha': alpha, 'checkpoints': checkpoints})
    if 'score_matrix' in inspect.signature(choose_tau).parameters:
        candidates.append({'score_matrix': val_scores, 'alpha': alpha, 'checkpoints': checkpoints})
    if 'val_scores' in inspect.signature(choose_tau).parameters:
        candidates.append({'val_scores': val_scores, 'alpha': alpha, 'checkpoints': checkpoints})
    if 'scores' in inspect.signature(choose_tau).parameters:
        candidates.append({'scores': val_scores, 'alpha': alpha, 'checkpoints': checkpoints})
    if not candidates:
        candidates.append({'scores': val_scores, 'alpha': alpha, 'checkpoints': checkpoints})

    last_error = None
    for kwargs in candidates:
        try:
            tau = choose_tau(**kwargs)
            return float(np.asarray(tau).ravel()[0]), None
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            raise
    if last_error is not None:
        raise TypeError(f'Unable to call choose_tau for {model_name}: {last_error}')
    return None, 'Tau unavailable: plotting without tau.'


def _resolve_sample_id(meta, sample_id):
    if sample_id is not None:
        if sample_id not in meta['sample_id'].tolist():
            raise ValueError(f'Unknown sample_id {sample_id!r} in {meta.shape[0]} rows of meta.csv')
        return sample_id
    return _default_sample_id(meta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Processed data directory')
    parser.add_argument(
        '--scores-dir',
        default='results/scores',
        help='Directory with model score files (default: results/scores)',
    )
    parser.add_argument('--models', nargs='+', default=['bag', 'gru'], help='Model names to plot')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the score file name')
    parser.add_argument('--sample-id', default=None, help='Trace sample id to plot')
    parser.add_argument('--out', default='results/figures/replay.png', help='Output PNG path')
    parser.add_argument(
        '--tau-from-val',
        action='store_true',
        help='Draw the alert threshold computed from validation benign traces.',
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    meta = pd.read_csv(data_dir / 'meta.csv')
    sample_id = _resolve_sample_id(meta, args.sample_id)
    checkpoints = json.loads((data_dir / 'checkpoints.json').read_text())

    trace = np.load(data_dir / 'traces' / f'{sample_id}.npz', allow_pickle=False)
    tok = trace['tok'].astype(np.int32)
    dmg = trace['dmg'].astype(np.int32)
    damage_cumsum = np.cumsum(dmg, dtype=np.int64)

    scores_by_model = {}
    tau_by_model = {}
    for model_name in args.models:
        score_path = Path(args.scores_dir) / f'{model_name}_test_seed{args.seed}.npz'
        payload = _load_score_payload(score_path)
        sample_ids = np.asarray(payload['sample_ids'])
        if sample_id not in sample_ids.tolist():
            raise ValueError(
                f"Sample id {sample_id!r} is missing from {score_path}; available ids are: {sample_ids.tolist()[:5]}"
            )
        file_checkpoints = np.asarray(payload['checkpoints'])
        if not np.array_equal(file_checkpoints, checkpoints):
            raise ValueError(
                f'{model_name} checkpoint mismatch: score file checkpoints={file_checkpoints.tolist()} ; '
                f'expected={checkpoints}'
            )
        row_index = np.where(sample_ids == sample_id)[0][0]
        scores = np.asarray(payload['scores'][row_index], dtype=np.float32)
        scores_by_model[model_name] = scores
        if args.tau_from_val:
            tau, message = _compute_tau(
                model_name=model_name,
                scores_dir=args.scores_dir,
                data_dir=data_dir,
                seed=args.seed,
                checkpoints=checkpoints,
                meta=meta,
                alpha=0.05,
            )
            if message is not None:
                print(message)
            tau_by_model[model_name] = tau

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = 'Replay demo'
    if 'stub' in str(data_dir).lower():
        title = 'Replay demo - STUB DATA'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax1.set_title(title)
    ax1.set_xlabel('Events seen')
    ax1.set_ylabel('Ransomware score')
    ax1.grid(alpha=0.25)

    for model_name, scores in scores_by_model.items():
        ax1.plot(checkpoints, scores, marker='o', label=model_name)
        tau = tau_by_model.get(model_name)
        if tau is not None:
            ax1.axhline(tau, linestyle='--', alpha=0.8, label=f'{model_name} tau={tau:.3f}')
            alert_index = next(
                (i for i, value in enumerate(scores) if value >= tau),
                None,
            )
            if alert_index is not None:
                alert_checkpoint = int(checkpoints[alert_index])
                alert_score = float(scores[alert_index])
                ax1.scatter([alert_checkpoint], [alert_score], s=50, zorder=3)

    ax1.legend(loc='best')

    ax2.set_xlabel('Events seen')
    ax2.set_ylabel('Cumulative damage')
    ax2.grid(alpha=0.25)
    ax2.plot(np.arange(1, len(dmg) + 1), damage_cumsum, color='black', label='cumulative damage')

    for model_name, scores in scores_by_model.items():
        tau = tau_by_model.get(model_name)
        if tau is None:
            continue
        alert_index = next((i for i, value in enumerate(scores) if value >= tau), None)
        if alert_index is None:
            continue
        alert_checkpoint = int(checkpoints[alert_index])
        damage_at_alert = int(damage_cumsum[min(alert_checkpoint, len(damage_cumsum) - 1)])
        ax2.scatter([alert_checkpoint], [damage_at_alert], s=50, label=f'{model_name} alert')

    ax2.legend(loc='best')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'Replay plot written to: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
