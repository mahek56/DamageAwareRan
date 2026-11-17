"""Threshold selection from validation benign trace scores."""

import numpy as np


def max_scores(scores):
    """Return each trace's maximum checkpoint score as float64."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError('scores must have shape [traces, checkpoints]')
    if values.shape[1] == 0:
        raise ValueError('scores must contain at least one checkpoint')
    if not np.all(np.isfinite(values)):
        raise ValueError('scores must contain only finite values')
    return np.max(values, axis=1)


def choose_tau(benign_max, alpha):
    """Choose the smallest threshold meeting the validation benign FPR limit."""
    values = np.asarray(benign_max, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('benign_max must be one-dimensional')
    if values.size == 0:
        raise ValueError('benign_max must contain at least one trace')
    if not np.all(np.isfinite(values)):
        raise ValueError('benign_max must contain only finite values')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1')

    max_flagged = int(np.floor(alpha * len(values) + 1e-9))
    candidates = np.append(
        np.unique(values),
        np.nextafter(values.max(), np.inf),
    )
    for candidate in candidates:
        if np.count_nonzero(values >= candidate) <= max_flagged:
            return np.float64(candidate)
    raise RuntimeError('No threshold satisfies the requested alpha')
