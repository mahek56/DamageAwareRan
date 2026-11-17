from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from damageaware.contracts import (
    LABELS,
    load_meta,
    load_scores,
    load_trace,
    save_scores,
)
from scripts.run_model import MODEL_REGISTRY


def _train_ids_by_label(meta, n_per_label=2):
    ids = []
    for label in LABELS:
        label_ids = meta.loc[
            (meta['label'] == label) & (meta['split'] == 'train'),
            'sample_id',
        ].tolist()
        ids.extend(label_ids[:n_per_label])
    return ids


def _val_ids_by_label(meta, n_per_label=2):
    ids = []
    for label in LABELS:
        label_ids = meta.loc[
            (meta['label'] == label) & (meta['split'] == 'val'),
            'sample_id',
        ].tolist()
        ids.extend(label_ids[:n_per_label])
    return ids


def test_registry_models_are_causal(data_dir):
    meta = load_meta(data_dir)
    train_ids = _train_ids_by_label(meta)
    val_ids = _val_ids_by_label(meta)
    if not train_ids or not val_ids:
        pytest.skip('not enough labeled rows to run causal checks')

    for name, (module_path, class_name) in MODEL_REGISTRY.items():
        try:
            module = import_module(module_path)
        except ModuleNotFoundError:
            continue

        detector_class = getattr(module, class_name, None)
        if detector_class is None:
            continue

        detector = detector_class(max_epochs=1) if name == 'gru' else detector_class()
        detector.fit(train_ids, val_ids, data_dir, seed=0)
        sample_id = meta.loc[meta['n_events'].idxmax(), 'sample_id']
        tok = load_trace(data_dir, sample_id)['tok']
        checkpoints = detector.checkpoints[: min(4, len(detector.checkpoints))]
        if not checkpoints:
            continue

        from tests.helpers import assert_prefix_causal

        assert_prefix_causal(detector, tok, checkpoints, np.random.default_rng(0))
        scores = detector.score_prefixes(tok, checkpoints)
        assert np.shape(scores) == (len(checkpoints),)


def test_no_cluster_crosses_splits(data_dir):
    meta = load_meta(data_dir)
    required = ['sample_id', 'label', 'split', 'split_random', 'cluster_id', 'n_events', 'total_damage_full']
    missing = [column for column in required if column not in meta.columns]
    assert not missing, f'Missing required metadata columns: {missing}'
    assert (meta.groupby('cluster_id')['split'].nunique() == 1).all()


def test_train_only_vocab_test_is_skipped_until_r1_build_script_exists(data_dir):
    del data_dir
    build_script = Path('scripts/build_data.py')
    if not build_script.exists():
        pytest.skip('R1 build script is not present yet; train-only vocab validation waits for that step.')


def test_determinism_for_bag_and_gru(data_dir):
    meta = load_meta(data_dir)
    train_ids = _train_ids_by_label(meta)
    val_ids = _val_ids_by_label(meta)
    if not train_ids or not val_ids:
        pytest.skip('not enough labeled rows to run determinism checks')

    sample_id = meta.loc[meta['n_events'].idxmax(), 'sample_id']
    tok = load_trace(data_dir, sample_id)['tok']
    checkpoints = [8, 16, 32]

    for name in ('bag', 'gru'):
        if name not in MODEL_REGISTRY:
            continue
        module_path, class_name = MODEL_REGISTRY[name]
        try:
            module = import_module(module_path)
        except ModuleNotFoundError:
            continue

        detector_class = getattr(module, class_name, None)
        if detector_class is None:
            continue

        det_a = detector_class(max_epochs=1) if name == 'gru' else detector_class()
        det_a.fit(train_ids, val_ids, data_dir, seed=0)
        scores_a = det_a.score_prefixes(tok, checkpoints)

        det_b = detector_class(max_epochs=1) if name == 'gru' else detector_class()
        det_b.fit(train_ids, val_ids, data_dir, seed=0)
        scores_b = det_b.score_prefixes(tok, checkpoints)
        assert np.array_equal(scores_a, scores_b)

        if name == 'gru':
            det_c = detector_class(max_epochs=1)
            det_c.fit(train_ids, val_ids, data_dir, seed=1)
            scores_c = det_c.score_prefixes(tok, checkpoints)
            assert not np.allclose(scores_a, scores_c, atol=1e-6)


def test_contracts_roundtrip_and_reject_bad_shapes(tmp_path):
    sample_ids = ['s0', 's1', 's2']
    checkpoints = [8, 16]
    scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    final_probs = np.full((3, len(LABELS)), 1.0 / len(LABELS), dtype=np.float32)
    path = tmp_path / 'scores.npz'

    save_scores(path, sample_ids, checkpoints, scores, final_probs=final_probs)
    payload = load_scores(path)
    assert payload['sample_ids'].tolist() == sample_ids
    assert payload['checkpoints'].tolist() == checkpoints
    assert payload['scores'].shape == (3, 2)
    assert payload['final_probs'].shape == (3, len(LABELS))

    with pytest.raises(ValueError):
        save_scores(path, sample_ids, checkpoints, np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        save_scores(
            path,
            sample_ids,
            checkpoints,
            scores,
            final_probs=np.ones((2, len(LABELS)), dtype=np.float32),
        )
