"""Train a detector and write validation and test prefix scores."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import sklearn
from sklearn.metrics import f1_score

from damageaware.contracts import LABELS, load_json, load_meta, load_trace, save_scores


MODEL_REGISTRY = {
    'bag': ('damageaware.models.bag', 'BagOfAPIsDetector'),
    'gru': ('damageaware.models.gru', 'GRUDetector'),
}
# R3 adds 'cnn', R2 adds 'gru'.


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Detector name')
    parser.add_argument('--data-dir', required=True, help='Processed data directory')
    parser.add_argument('--seed', required=True, type=int, help='Random seed')
    parser.add_argument(
        '--out-dir',
        default='results/scores',
        help='Directory for score files (default: results/scores)',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Maximum GRU training epochs (default: 30)',
    )
    args = parser.parse_args(argv)
    if args.model not in MODEL_REGISTRY:
        registered = ', '.join(sorted(MODEL_REGISTRY))
        parser.error(f"unknown --model {args.model!r}; registered models: {registered}")
    if args.max_epochs is not None:
        if not 1 <= args.max_epochs <= 30:
            parser.error('--max-epochs must be between 1 and 30')
        if args.model != 'gru':
            parser.error('--max-epochs is only supported for the gru model')
    return args


def _create_detector(name, max_epochs=None):
    if name not in MODEL_REGISTRY:
        registered = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model {name!r}; registered models: {registered}")
    module_path, class_name = MODEL_REGISTRY[name]
    model_module = import_module(module_path)
    detector_class = getattr(model_module, class_name)
    if name == 'gru' and max_epochs is not None:
        return detector_class(max_epochs=max_epochs)
    return detector_class()


def _seed_libraries(seed):
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != 'torch':
            raise
    else:
        torch.manual_seed(seed)


def _score_split(detector, sample_ids, data_dir, checkpoints):
    scores = np.empty((len(sample_ids), len(checkpoints)), dtype=np.float32)
    final_probs = np.empty((len(sample_ids), len(LABELS)), dtype=np.float32)
    for row_index, sample_id in enumerate(sample_ids):
        tok = load_trace(data_dir, sample_id)['tok']
        scores[row_index] = detector.score_prefixes(tok, checkpoints)
        final_probs[row_index] = detector.final_probs(tok)
    return scores, final_probs


def main(argv=None):
    args = _parse_args(argv)
    _seed_libraries(args.seed)

    print(f'seed: {args.seed}')
    print(f'python: {sys.version.split()[0]}')
    print(f'numpy: {np.__version__}')
    print(f'sklearn: {sklearn.__version__}')

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)
    train_ids = meta.loc[meta['split'] == 'train', 'sample_id'].tolist()
    val_meta = meta.loc[meta['split'] == 'val']
    val_ids = val_meta['sample_id'].tolist()

    detector = _create_detector(args.model, args.max_epochs)
    detector.fit(train_ids, val_ids, data_dir, args.seed)
    checkpoints = load_json(data_dir, 'checkpoints.json')

    val_scores, val_final_probs = _score_split(
        detector, val_ids, data_dir, checkpoints
    )
    out_dir = Path(args.out_dir)
    save_scores(
        out_dir / f'{args.model}_val_seed{args.seed}.npz',
        val_ids,
        checkpoints,
        val_scores,
        final_probs=val_final_probs,
    )
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    val_labels = [label_to_index[label] for label in val_meta['label']]
    val_predictions = np.argmax(val_final_probs, axis=1)
    val_macro_f1 = f1_score(
        val_labels,
        val_predictions,
        labels=range(len(LABELS)),
        average='macro',
        zero_division=0,
    )
    print(f'val macro-F1: {val_macro_f1:.6f}')

    test_ids = meta.loc[meta['split'] == 'test', 'sample_id'].tolist()
    test_scores, test_final_probs = _score_split(
        detector, test_ids, data_dir, checkpoints
    )
    save_scores(
        out_dir / f'{args.model}_test_seed{args.seed}.npz',
        test_ids,
        checkpoints,
        test_scores,
        final_probs=test_final_probs,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
