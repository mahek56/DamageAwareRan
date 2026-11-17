import numpy as np


def assert_prefix_causal(detector, tok, checkpoints, rng):
    tok = np.asarray(tok, dtype=np.int64)
    checkpoints = sorted(int(checkpoint) for checkpoint in checkpoints)
    if not checkpoints:
        return

    upper_bound = int(getattr(detector, 'vocab_size', max(tok.max(), 1) + 1))
    valid_ids = np.arange(1, upper_bound, dtype=np.int64)
    if valid_ids.size == 0:
        valid_ids = np.array([1], dtype=np.int64)

    for checkpoint in checkpoints:
        if checkpoint <= 0 or checkpoint >= len(tok):
            continue
        mutated = tok.copy()
        mutated[checkpoint:] = rng.choice(valid_ids, size=len(mutated) - checkpoint, replace=True)

        before = detector.score_prefixes(tok, [c for c in checkpoints if c <= checkpoint])
        after = detector.score_prefixes(mutated, [c for c in checkpoints if c <= checkpoint])
        assert np.allclose(before, after, atol=1e-6)
