import numpy as np
import pandas as pd
import pytest

from damageaware.eval.damage import alert_index, evaluate_trace
from damageaware.eval.metrics import compute_metrics
from damageaware.eval.thresholds import choose_tau, max_scores
from damageaware.contracts import load_json
from scripts.evaluate import _load_aligned_scores, main


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


def _tiny_metrics_inputs():
    return {
        'model': 'toy',
        'alpha': 0.5,
        'seed': 4,
        'val_scores': np.array(
            [[0.1, 0.2, 0.1], [0.1, 0.3, 0.2]],
            dtype=np.float64,
        ),
        'val_labels': ['benign', 'benign'],
        'test_scores': np.array(
            [
                [0.3, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.6, 0.1],
                [0.1, 0.2, 0.1],
                [0.9, 0.9, 0.9],
                [0.4, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
        'test_labels': [
            'benign',
            'benign',
            'ransomware',
            'ransomware',
            'ransomware',
            'rat',
        ],
        'test_damages': [
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 2, 1]),
            np.array([0, 1, 0, 2, 1]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ],
        'test_n_events': np.array([5, 5, 5, 5, 5, 5]),
        'test_total_damage': np.array([0, 0, 4, 4, 0, 0]),
        'checkpoints': np.array([2, 4, 8]),
        'bootstrap_seed': 91,
        'n_bootstrap': 200,
    }


def test_metrics_match_hand_computed_case():
    row, trace_results = compute_metrics(**_tiny_metrics_inputs())

    assert row['tau'] == 0.3
    assert row['realized_fpr_benign'] == 0.5
    assert row['realized_fpr_nonransomware'] == pytest.approx(2 / 3)
    assert row['recall'] == pytest.approx(2 / 3)
    assert row['median_damage'] == 3
    assert row['p90_damage'] == pytest.approx(3.8)
    assert row['frac_never_caught'] == pytest.approx(1 / 3)
    assert row['frac_zero_damage'] == pytest.approx(1 / 3)
    assert row['median_trace_fraction_at_alert'] == pytest.approx(0.8)
    assert [result['damage'] for result in trace_results['trace_results']] == [3, 4, 0]


def test_bootstrap_is_deterministic_and_interval_contains_point():
    inputs = _tiny_metrics_inputs()
    first, _ = compute_metrics(**inputs)
    second, _ = compute_metrics(**inputs)

    assert first['recall_ci_low'] == second['recall_ci_low']
    assert first['recall_ci_high'] == second['recall_ci_high']
    assert first['recall_ci_low'] <= first['recall'] <= first['recall_ci_high']


def test_evaluate_main_writes_stub_tables(tmp_path):
    out_dir = tmp_path / 'evaluation'
    assert main(
        [
            '--data-dir',
            'data/processed_stub',
            '--models',
            'bag',
            'gru',
            '--alphas',
            '0.05',
            '--seeds',
            '0',
            '--scores-dir',
            'results/scores',
            '--out-dir',
            str(out_dir),
        ]
    ) == 0

    main_table = pd.read_csv(out_dir / 'tables' / 'main.csv')
    expected_columns = {
        'model',
        'alpha',
        'seed',
        'tau',
        'realized_fpr_benign',
        'realized_fpr_nonransomware',
        'recall',
        'median_damage',
        'p90_damage',
        'frac_zero_damage',
        'frac_never_caught',
        'median_trace_fraction_at_alert',
        'macro_f1_4class',
        'ranking_kendall_tau',
    }
    assert expected_columns.issubset(main_table.columns)
    assert len(main_table) == 2
    assert set(main_table['model']) == {'bag', 'gru'}
    assert set(main_table['alpha']) == {0.05}
    assert set(main_table['seed']) == {0}
    assert (out_dir / 'tables' / 'main_summary.csv').is_file()
    metadata = pd.read_json(out_dir / 'tables' / 'main.meta.json', typ='series')
    assert {
        'command_line',
        'config_contents',
        'git_commit_hash',
        'dirty',
        'damage_level',
        'library_versions',
        'timestamp',
    }.issubset(metadata.index)
    assert list((out_dir / 'figures').glob('*.png'))


def test_aligned_score_loader_reorders_by_sample_id(tmp_path):
    path = tmp_path / 'scores.npz'
    np.savez(
        path,
        sample_ids=np.array(['second', 'first']),
        checkpoints=np.array([8, 16]),
        scores=np.array([[0.8, 0.9], [0.1, 0.2]], dtype=np.float32),
    )

    loaded = _load_aligned_scores(path, ['first', 'second'], np.array([8, 16]))

    assert loaded['sample_ids'] == ['first', 'second']
    assert loaded['scores'].dtype == np.float64
    expected = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32).astype(np.float64)
    assert np.array_equal(loaded['scores'], expected)


def test_aligned_score_loader_rejects_checkpoint_mismatch(tmp_path):
    path = tmp_path / 'scores.npz'
    np.savez(
        path,
        sample_ids=np.array(['sample']),
        checkpoints=np.array([8, 24]),
        scores=np.array([[0.1, 0.2]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match='Checkpoint mismatch'):
        _load_aligned_scores(path, ['sample'], np.array([8, 16]))


def test_aligned_score_loader_rejects_sample_id_mismatch(tmp_path):
    path = tmp_path / 'scores.npz'
    np.savez(
        path,
        sample_ids=np.array(['unexpected']),
        checkpoints=np.array([8]),
        scores=np.array([[0.1]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match='Sample IDs.*do not match'):
        _load_aligned_scores(path, ['expected'], np.array([8]))
