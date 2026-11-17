"""Detector interface. score_prefixes at checkpoint c may use ONLY tok[:c]."""
import numpy as np
class Detector:
    name = 'base'
    def fit(self, train_ids, val_ids, data_dir, seed): raise NotImplementedError
    def score_prefixes(self, tok, checkpoints):
        """Return np.ndarray[len(checkpoints)]: P(ransomware | first min(c, len(tok)) events)."""
        raise NotImplementedError
    def final_probs(self, tok):
        """Return np.ndarray[4] class probabilities on the whole (capped) trace."""
        raise NotImplementedError
