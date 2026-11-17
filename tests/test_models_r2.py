from pathlib import Path

import numpy as np
import pytest

from damageaware.contracts import (
    LABELS,
    load_json,
    load_meta,
    load_scores,
    load_trace,
)
from damageaware.models.bag import BagOfAPIsDetector
from scripts.run_model import main


DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'processed_stub'


@pytest.fixture(scope='module')
def fitted_bag():
    meta = load_meta(DATA_DIR)
    train = meta[meta['split'] == 'train']
    train_ids = [
        sample_id
        for label in LABELS
        for sample_id in train.loc[train['label'] == label, 'sample_id'].head(2)
    ]
    val_ids = meta.loc[meta['split'] == 'val', 'sample_id'].tolist()
    detector = BagOfAPIsDetector().fit(train_ids, val_ids, DATA_DIR, seed=17)
    sample_id = train.loc[train['n_events'].idxmax(), 'sample_id']
    tok = load_trace(DATA_DIR, sample_id)['tok']
    return detector, tok


def test_bag_output_shapes(fitted_bag):
    detector, tok = fitted_bag
    checkpoints = detector.checkpoints[:3]

    scores = detector.score_prefixes(tok, checkpoints)
    final_probs = detector.final_probs(tok)

    assert scores.shape == (len(checkpoints),)
    assert final_probs.shape == (4,)
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(final_probs))
    assert np.isclose(final_probs.sum(), 1.0)


def test_bag_score_is_causal_at_checkpoint(fitted_bag):
    detector, tok = fitted_bag
    checkpoint = detector.checkpoints[0]
    assert len(tok) > checkpoint

    changed = tok.copy()
    changed[checkpoint:] = 3
    assert not np.array_equal(tok[checkpoint:], changed[checkpoint:])

    original_score = detector.score_prefixes(tok, [checkpoint])[0]
    changed_score = detector.score_prefixes(changed, [checkpoint])[0]
    assert changed_score == pytest.approx(original_score)


def test_bag_trains_on_stub_data(fitted_bag):
    detector, tok = fitted_bag

    assert detector.classifier is not None
    assert detector.classifier.max_iter == 1000
    assert detector.C in detector.c_grid
    assert detector.final_probs(tok).shape == (4,)


def test_bag_vocab_gap_and_max_token_id(fitted_bag):
    detector, _ = fitted_bag
    vocab = load_json(DATA_DIR, 'vocab.json')
    max_token_id = max(int(token_id) for token_id in vocab.values())

    assert len(vocab) < max_token_id + 1
    assert detector.vocab_size == max_token_id + 1
    scores = detector.score_prefixes(np.array([max_token_id]), [8])
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])


def test_bag_handles_short_and_empty_traces(fitted_bag):
    detector, _ = fitted_bag
    short_tok = np.array([3, 4, 5])

    short_score = detector.score_prefixes(short_tok, [8])
    exact_score = detector.score_prefixes(short_tok, [len(short_tok)])
    short_probs = detector.final_probs(short_tok)
    empty_score = detector.score_prefixes(np.array([], dtype=np.int32), [8])
    empty_probs = detector.final_probs(np.array([], dtype=np.int32))

    assert short_score == pytest.approx(exact_score)
    assert short_probs.shape == (4,)
    assert empty_score.shape == (1,)
    assert empty_probs.shape == (4,)
    assert np.all(np.isfinite(empty_score))
    assert np.all(np.isfinite(empty_probs))
    assert np.array_equal(
        detector._feature(np.array([], dtype=np.int32), 8),
        np.zeros(detector.vocab_size),
    )


def test_run_model_writes_bag_val_and_test_scores(tmp_path, capsys):
    out_dir = tmp_path / 'scores'
    assert main(
        [
            '--model',
            'bag',
            '--data-dir',
            str(DATA_DIR),
            '--seed',
            '0',
            '--out-dir',
            str(out_dir),
        ]
    ) == 0
    assert 'val macro-F1:' in capsys.readouterr().out

    meta = load_meta(DATA_DIR)
    checkpoints = load_json(DATA_DIR, 'checkpoints.json')
    for split in ('val', 'test'):
        expected_ids = meta.loc[meta['split'] == split, 'sample_id'].tolist()
        scores = load_scores(out_dir / f'bag_{split}_seed0.npz')
        assert scores['scores'].shape == (len(expected_ids), len(checkpoints))
        assert scores['final_probs'].shape == (len(expected_ids), 4)
        assert scores['sample_ids'].tolist() == expected_ids
        assert np.array_equal(scores['checkpoints'], checkpoints)


def test_run_model_rejects_unknown_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                '--model',
                'unknown',
                '--data-dir',
                str(DATA_DIR),
                '--seed',
                '0',
                '--out-dir',
                str(tmp_path),
            ]
        )

    assert error.value.code == 2
    assert 'registered models: bag' in capsys.readouterr().err
