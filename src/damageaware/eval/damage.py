"""Per-trace alert and damage calculations.

Assumption: a trace never caught at any checkpoint has trace_fraction 1.0.
"""

import numpy as np


def alert_index(score_row, tau):
    """Return the index of the first checkpoint score at or above tau."""
    scores = np.asarray(score_row)
    if scores.ndim != 1:
        raise ValueError('score_row must be one-dimensional')
    alerted = np.flatnonzero(scores >= tau)
    return int(alerted[0]) if alerted.size else None


def evaluate_trace(score_row, checkpoints, tau, dmg, n_events, total_damage_full):
    """Return caught status, alert checkpoint, damage, and trace fraction."""
    scores = np.asarray(score_row)
    checkpoint_values = np.asarray(checkpoints)
    damage_values = np.asarray(dmg)
    if scores.ndim != 1 or checkpoint_values.ndim != 1:
        raise ValueError('score_row and checkpoints must be one-dimensional')
    if scores.size != checkpoint_values.size:
        raise ValueError('score_row and checkpoints must have equal lengths')
    if damage_values.ndim != 1:
        raise ValueError('dmg must be one-dimensional')
    if int(n_events) != n_events or n_events < 0:
        raise ValueError('n_events must be a non-negative integer')
    if np.any(checkpoint_values <= 0):
        raise ValueError('checkpoints must be positive')

    index = alert_index(scores, tau)
    if index is None:
        return {
            'caught': False,
            'alert_checkpoint': None,
            'damage': total_damage_full,
            'trace_fraction': 1.0,
        }

    checkpoint = int(checkpoint_values[index])
    events_seen = min(checkpoint, int(n_events))
    if not damage_values.size or events_seen == 0:
        damage_at_alert = 0
    else:
        if damage_values.size < int(n_events):
            raise ValueError('dmg must cover n_events when non-empty')
        cumulative_damage = np.cumsum(damage_values)
        damage_at_alert = cumulative_damage[events_seen - 1].item()
    trace_fraction = events_seen / n_events if n_events else 0.0
    return {
        'caught': True,
        'alert_checkpoint': checkpoint,
        'damage': damage_at_alert,
        'trace_fraction': trace_fraction,
    }
