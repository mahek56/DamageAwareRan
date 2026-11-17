import numpy as np
import pytest

from damageaware.eval.damage import alert_index, evaluate_trace
from damageaware.eval.thresholds import choose_tau, max_scores


def test_max_scores_returns_float64_per_trace_maximum():
    scores = np.array([[0.1, 0.4, 0.2], [0.8, 0.3, 0.5]], dtype=np.float32)

    result = max_scores(scores)

    assert result.dtype == np.float64
    expected = np.array([0.4, 0.8], dtype=np.float32).astype(np.float64)
    assert np.array_equal(result, expected)


def test_choose_tau_uses_smallest_candidate_that_meets_alpha():
    benign_max = np.array([0.1, 0.2, 0.3, 0.9])

    assert choose_tau(benign_max, 0.25) == 0.9
    assert choose_tau(benign_max, 0.5) == 0.3
    assert choose_tau(benign_max, 0.05) == np.nextafter(0.9, np.inf)


def test_choose_tau_ties_use_greater_than_or_equal():
    benign_max = np.array([0.2, 0.2, 0.5, 0.5])

    assert choose_tau(benign_max, 0.5) == 0.5


def test_choose_tau_rejects_empty_benign_set():
    with pytest.raises(ValueError, match='at least one trace'):
        choose_tau(np.array([], dtype=np.float64), 0.0)


def test_alert_index_returns_first_checkpoint():
    assert alert_index(np.array([0.6, 0.8, 0.9]), 0.5) == 0
    assert alert_index(np.array([0.1, 0.2, 0.4]), 0.5) is None


def test_evaluate_trace_computes_damage_at_alert():
    result = evaluate_trace(
        score_row=np.array([0.1, 0.6, 0.9]),
        checkpoints=np.array([2, 4, 8]),
        tau=0.5,
        dmg=np.array([0, 1, 0, 2, 1]),
        n_events=5,
        total_damage_full=4,
    )

    assert result == {
        'caught': True,
        'alert_checkpoint': 4,
        'damage': 3,
        'trace_fraction': 0.8,
    }


def test_evaluate_trace_never_caught_uses_full_damage_and_fraction_one():
    result = evaluate_trace(
        score_row=np.array([0.1, 0.2, 0.4]),
        checkpoints=np.array([2, 4, 8]),
        tau=0.5,
        dmg=np.array([0, 1, 0, 2, 1]),
        n_events=5,
        total_damage_full=4,
    )

    assert result == {
        'caught': False,
        'alert_checkpoint': None,
        'damage': 4,
        'trace_fraction': 1.0,
    }


def test_evaluate_trace_caps_larger_checkpoint_at_event_count():
    result = evaluate_trace(
        score_row=np.array([0.9]),
        checkpoints=np.array([8]),
        tau=0.5,
        dmg=np.array([0, 1, 0, 2, 1]),
        n_events=5,
        total_damage_full=4,
    )

    assert result['caught'] is True
    assert result['alert_checkpoint'] == 8
    assert result['damage'] == 4
    assert result['trace_fraction'] == 1.0


def test_evaluate_trace_handles_empty_damage():
    result = evaluate_trace(
        score_row=np.array([0.9]),
        checkpoints=np.array([2]),
        tau=0.5,
        dmg=np.array([], dtype=np.int32),
        n_events=5,
        total_damage_full=4,
    )

    assert result['caught'] is True
    assert result['damage'] == 0
    assert result['trace_fraction'] == 0.4
