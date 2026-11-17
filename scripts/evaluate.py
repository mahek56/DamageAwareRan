"""Evaluate saved detector score files against processed trace data."""

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from damageaware.contracts import LABELS, load_json, load_meta, load_scores, load_trace
from damageaware.eval.figures import write_figures
from damageaware.eval.metrics import add_rankings, compute_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'default.yaml'


def _read_config(path):
    with path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def _parser(config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=config.get('data_dir'))
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument(
        '--alphas',
        nargs='+',
        type=float,
        default=config.get('alphas', [0.05, 0.10]),
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=config.get('seeds', [0, 1, 2]),
    )
    parser.add_argument('--scores-dir', default='results/scores')
    parser.add_argument('--out-dir', default='results')
    return parser


def _load_aligned_scores(path, expected_ids, expected_checkpoints):
    if not path.is_file():
        raise FileNotFoundError(f'Missing score file: {path}')
    payload = load_scores(path)
    if 'sample_ids' not in payload or 'checkpoints' not in payload or 'scores' not in payload:
        raise ValueError(f'Score file {path} is missing required arrays')

    stored_ids = [str(sample_id) for sample_id in payload['sample_ids'].tolist()]
    expected_ids = [str(sample_id) for sample_id in expected_ids]
    if len(stored_ids) != len(set(stored_ids)):
        raise ValueError(f'Score file {path} contains duplicate sample_ids')
    if len(stored_ids) != len(expected_ids) or set(stored_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(stored_ids))
        extra = sorted(set(stored_ids) - set(expected_ids))
        raise ValueError(
            f'Sample IDs in {path} do not match meta split; '
            f'missing={missing[:5]}, extra={extra[:5]}'
        )
    if not np.array_equal(payload['checkpoints'], expected_checkpoints):
        raise ValueError(f'Checkpoint mismatch in score file {path}')

    # Convert exactly once after loading; all thresholding and alert checks use this array.
    scores = np.asarray(payload['scores'], dtype=np.float64)
    if scores.ndim != 2 or scores.shape != (len(stored_ids), len(expected_checkpoints)):
        raise ValueError(f'Invalid scores shape in {path}: {scores.shape}')
    order = {sample_id: index for index, sample_id in enumerate(stored_ids)}
    row_order = [order[sample_id] for sample_id in expected_ids]
    aligned = {'sample_ids': expected_ids, 'scores': scores[row_order]}
    if 'final_probs' in payload:
        final_probs = np.asarray(payload['final_probs'], dtype=np.float64)
        if final_probs.shape != (len(stored_ids), len(LABELS)):
            raise ValueError(f'Invalid final_probs shape in {path}: {final_probs.shape}')
        aligned['final_probs'] = final_probs[row_order]
    return aligned


def _git_metadata():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return 'unknown', None
    return commit, dirty


def _save_metadata(out_dir, command_line, config, config_path):
    commit, dirty = _git_metadata()
    metadata = {
        'command_line': command_line,
        'config_contents': config,
        'config_path': str(config_path),
        'git_commit_hash': commit,
        'dirty': dirty,
        'damage_level': config.get('damage_level'),
        'library_versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__,
            'scipy': scipy.__version__,
            'matplotlib': matplotlib.__version__,
        },
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = out_dir / 'tables' / 'main.meta.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open('w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write('\n')
    return metadata_path


def _summary(frame):
    excluded = {
        'model',
        'alpha',
        'seed',
        'macro_f1_note',
        'rankings_agree',
    }
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    grouped = frame.groupby(['model', 'alpha'], as_index=False, sort=False)
    means = grouped[numeric].mean().rename(
        columns={column: f'{column}_mean' for column in numeric}
    )
    stds = frame.groupby(['model', 'alpha'], as_index=False, sort=False)[numeric].std(
        ddof=1
    ).rename(columns={column: f'{column}_std' for column in numeric})
    return means.merge(stds, on=['model', 'alpha'], how='left')


def main(argv=None):
    config_path = DEFAULT_CONFIG
    config = _read_config(config_path)
    args = _parser(config).parse_args(argv)
    if not args.data_dir:
        raise ValueError('--data-dir is required when config has no data_dir')
    if not args.seeds or not args.alphas or not args.models:
        raise ValueError('models, alphas, and seeds must be non-empty')

    data_dir = Path(args.data_dir)
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    meta = load_meta(data_dir)
    if 'split' not in meta.columns or 'sample_id' not in meta.columns:
        raise ValueError('meta.csv must contain split and sample_id columns')
    if meta['sample_id'].astype(str).duplicated().any():
        raise ValueError('meta.csv contains duplicate sample_ids')
    checkpoints = np.asarray(load_json(data_dir, 'checkpoints.json'))
    val_meta = meta.loc[meta['split'] == 'val'].copy()
    test_meta = meta.loc[meta['split'] == 'test'].copy()
    val_ids = val_meta['sample_id'].astype(str).tolist()
    test_ids = test_meta['sample_id'].astype(str).tolist()

    rows = []
    trace_cache = {}
    bootstrap_seed = int(config.get('seeds', args.seeds)[0])
    for model in args.models:
        for seed in args.seeds:
            val_path = scores_dir / f'{model}_val_seed{seed}.npz'
            test_path = scores_dir / f'{model}_test_seed{seed}.npz'
            val_scores = _load_aligned_scores(val_path, val_ids, checkpoints)
            test_scores = _load_aligned_scores(test_path, test_ids, checkpoints)
            test_damages = [
                load_trace(data_dir, sample_id)['dmg']
                for sample_id in test_ids
            ]
            test_n_events = test_meta['n_events'].to_numpy()
            test_total_damage = test_meta['total_damage_full'].to_numpy()

            for alpha in args.alphas:
                row, traces = compute_metrics(
                    model=model,
                    alpha=alpha,
                    seed=seed,
                    val_scores=val_scores['scores'],
                    val_labels=val_meta['label'].tolist(),
                    test_scores=test_scores['scores'],
                    test_labels=test_meta['label'].tolist(),
                    test_damages=test_damages,
                    test_n_events=test_n_events,
                    test_total_damage=test_total_damage,
                    checkpoints=checkpoints,
                    bootstrap_seed=bootstrap_seed,
                    final_probs=test_scores.get('final_probs'),
                )
                rows.append(row)
                trace_cache[(model, float(alpha), int(seed))] = traces

    add_rankings(rows)
    out_dir.joinpath('tables').mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    main_path = out_dir / 'tables' / 'main.csv'
    summary_path = out_dir / 'tables' / 'main_summary.csv'
    frame.to_csv(main_path, index=False)
    _summary(frame).to_csv(summary_path, index=False)
    meta_path = _save_metadata(
        out_dir,
        [sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        config,
        config_path,
    )

    figure_data = {}
    figure_seed = 0 if 0 in args.seeds else None
    for alpha in args.alphas:
        figure_data[float(alpha)] = {}
        for model in args.models:
            selected_seeds = [figure_seed] if figure_seed is not None else args.seeds
            selected = [
                trace_cache[(model, float(alpha), int(seed))]
                for seed in selected_seeds
            ]
            damage = np.concatenate([trace['damage'] for trace in selected])
            fractions = np.concatenate([trace['trace_fraction'] for trace in selected])
            figure_data[float(alpha)][model] = {
                'damage': damage,
                'median_damage': float(np.median(damage)),
                'median_trace_fraction': float(np.median(fractions)),
            }
    figure_paths = write_figures(figure_data, out_dir / 'figures')

    columns = [
        'model',
        'alpha',
        'seed',
        'tau',
        'realized_fpr_benign',
        'recall',
        'median_damage',
        'p90_damage',
        'frac_never_caught',
        'median_trace_fraction_at_alert',
        'macro_f1_4class',
        'macro_f1_note',
    ]
    print(frame[columns].to_string(index=False))
    for row in rows:
        if row['macro_f1_note']:
            print(f"{row['model']} seed {row['seed']}: {row['macro_f1_note']}")
    print(f'Wrote: {main_path}')
    print(f'Wrote: {summary_path}')
    print(f'Wrote: {meta_path}')
    for path in figure_paths:
        print(f'Wrote: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
