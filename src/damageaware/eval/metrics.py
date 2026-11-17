"""Evaluation metrics derived from validation and test score arrays."""

import numpy as np
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import f1_score

from damageaware.contracts import LABELS
from damageaware.eval.damage import evaluate_trace
from damageaware.eval.thresholds import choose_tau, max_scores


def _label_ids(labels):
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    try:
        return np.asarray(
            [label_to_id[label] if isinstance(label, str) else int(label) for label in labels],
            dtype=np.int64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Unknown or invalid class label: {error}') from error


def _bootstrap_interval(values, statistic, rng, n_bootstrap):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('Bootstrap groups must be non-empty one-dimensional arrays')
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, values.size, size=values.size)
        samples[iteration] = statistic(values[indices])
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def _add_interval(row, name, values, statistic, rng, n_bootstrap):
    lower, upper = _bootstrap_interval(values, statistic, rng, n_bootstrap)
    row[f'{name}_ci_low'] = lower
    row[f'{name}_ci_high'] = upper


def compute_metrics(
    model,
    alpha,
    seed,
    val_scores,
    val_labels,
    test_scores,
    test_labels,
    test_damages,
    test_n_events,
    test_total_damage,
    checkpoints,
    bootstrap_seed,
    final_probs=None,
    n_bootstrap=1000,
):
    """Compute one model/alpha/seed row and its per-test-trace results."""
    val_scores = np.asarray(val_scores)
    test_scores = np.asarray(test_scores)
    val_label_ids = _label_ids(val_labels)
    test_label_ids = _label_ids(test_labels)
    checkpoints = np.asarray(checkpoints)

    if val_scores.ndim != 2 or val_scores.shape[0] != val_label_ids.size:
        raise ValueError('Validation scores and labels have inconsistent shapes')
    if test_scores.ndim != 2 or test_scores.shape[0] != test_label_ids.size:
        raise ValueError('Test scores and labels have inconsistent shapes')
    if val_scores.dtype != np.float64 or test_scores.dtype != np.float64:
        raise TypeError('Score arrays must be converted to float64 immediately after loading')
    if val_scores.shape[1] != len(checkpoints) or test_scores.shape[1] != len(checkpoints):
        raise ValueError('Score arrays and checkpoints have inconsistent shapes')
    if len(test_damages) != test_label_ids.size:
        raise ValueError('Test damages and labels have inconsistent lengths')
    if len(test_n_events) != test_label_ids.size:
        raise ValueError('Test n_events and labels have inconsistent lengths')
    if len(test_total_damage) != test_label_ids.size:
        raise ValueError('Test total damage and labels have inconsistent lengths')
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive')

    benign_val = val_label_ids == LABELS.index('benign')
    if not np.any(benign_val):
        raise ValueError('Validation split contains no benign traces for threshold selection')
    tau = choose_tau(max_scores(val_scores[benign_val]), alpha)

    flagged = np.any(test_scores >= tau, axis=1)
    benign_test = test_label_ids == LABELS.index('benign')
    nonransomware_test = test_label_ids != LABELS.index('ransomware')
    ransomware_test = test_label_ids == LABELS.index('ransomware')
    if not np.any(benign_test):
        raise ValueError('Test split contains no benign traces for realized FPR')
    if not np.any(nonransomware_test):
        raise ValueError('Test split contains no non-ransomware traces for realized FPR')
    if not np.any(ransomware_test):
        raise ValueError('Test split contains no ransomware traces for recall and damage')

    ransomware_indices = np.flatnonzero(ransomware_test)
    trace_results = [
        evaluate_trace(
            score_row=test_scores[index],
            checkpoints=checkpoints,
            tau=tau,
            dmg=test_damages[index],
            n_events=int(test_n_events[index]),
            total_damage_full=test_total_damage[index],
        )
        for index in ransomware_indices
    ]
    caught = np.asarray([result['caught'] for result in trace_results], dtype=bool)
    damage = np.asarray([result['damage'] for result in trace_results], dtype=np.float64)
    trace_fraction = np.asarray(
        [result['trace_fraction'] for result in trace_results],
        dtype=np.float64,
    )
    zero_damage = caught & (damage == 0)
    never_caught = ~caught

    rng = np.random.default_rng(bootstrap_seed)
    benign_flags = flagged[benign_test].astype(np.float64)
    nonransomware_flags = flagged[nonransomware_test].astype(np.float64)
    row = {
        'model': model,
        'alpha': float(alpha),
        'seed': int(seed),
        'tau': float(tau),
        'realized_fpr_benign': float(benign_flags.mean()),
        'realized_fpr_nonransomware': float(nonransomware_flags.mean()),
        'recall': float(caught.mean()),
        'median_damage': float(np.median(damage)),
        'p90_damage': float(np.percentile(damage, 90)),
        'frac_zero_damage': float(zero_damage.mean()),
        'frac_never_caught': float(never_caught.mean()),
        'median_trace_fraction_at_alert': float(np.median(trace_fraction)),
        'bootstrap_seed': int(bootstrap_seed),
    }
    _add_interval(
        row,
        'realized_fpr_benign',
        benign_flags,
        np.mean,
        rng,
        n_bootstrap,
    )
    _add_interval(
        row,
        'realized_fpr_nonransomware',
        nonransomware_flags,
        np.mean,
        rng,
        n_bootstrap,
    )
    _add_interval(row, 'recall', caught.astype(np.float64), np.mean, rng, n_bootstrap)
    _add_interval(row, 'median_damage', damage, np.median, rng, n_bootstrap)
    _add_interval(
        row,
        'p90_damage',
        damage,
        lambda values: np.percentile(values, 90),
        rng,
        n_bootstrap,
    )
    _add_interval(
        row,
        'frac_zero_damage',
        zero_damage.astype(np.float64),
        np.mean,
        rng,
        n_bootstrap,
    )
    _add_interval(
        row,
        'frac_never_caught',
        never_caught.astype(np.float64),
        np.mean,
        rng,
        n_bootstrap,
    )
    _add_interval(
        row,
        'median_trace_fraction_at_alert',
        trace_fraction,
        np.median,
        rng,
        n_bootstrap,
    )

    if final_probs is None:
        row['macro_f1_4class'] = np.nan
        row['macro_f1_note'] = 'skipped: final_probs absent'
    else:
        final_probs = np.asarray(final_probs)
        if final_probs.shape != (test_label_ids.size, len(LABELS)):
            raise ValueError('Test final_probs must have shape [traces, 4]')
        row['macro_f1_4class'] = float(
            f1_score(
                test_label_ids,
                np.argmax(final_probs, axis=1),
                labels=range(len(LABELS)),
                average='macro',
                zero_division=0,
            )
        )
        row['macro_f1_note'] = ''

    return row, {
        'caught': caught,
        'damage': damage,
        'trace_fraction': trace_fraction,
        'ransomware_indices': ransomware_indices,
        'trace_results': trace_results,
    }


def add_rankings(rows, minimum_models=3):
    """Add average-tie ranks and, when possible, Kendall ranking agreement."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row['alpha'], row['seed']), []).append(row)

    for group in grouped.values():
        damage_ranks = rankdata(
            [row['median_damage'] for row in group],
            method='average',
        )
        fraction_ranks = rankdata(
            [row['median_trace_fraction_at_alert'] for row in group],
            method='average',
        )
        for index, row in enumerate(group):
            row['median_damage_rank'] = float(damage_ranks[index])
            row['median_trace_fraction_rank'] = float(fraction_ranks[index])
            if len(group) < minimum_models:
                row['ranking_kendall_tau'] = np.nan
                row['rankings_agree'] = None
            else:
                row['ranking_kendall_tau'] = float(
                    kendalltau(damage_ranks, fraction_ranks).statistic
                )
                row['rankings_agree'] = bool(
                    np.array_equal(damage_ranks, fraction_ranks)
                )
