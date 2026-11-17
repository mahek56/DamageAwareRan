"""Bag-of-APIs detector trained on seeded random trace prefixes."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from damageaware.contracts import LABELS, load_json, load_meta, load_trace
from damageaware.models.base import Detector


class BagOfAPIsDetector(Detector):
    name = 'bag'
    n_prefixes_per_trace = 8
    c_grid = [1, 10, 100, 1000]

    def __init__(self):
        self.classifier = None
        self.checkpoints = None
        self.cap = None
        self.vocab_size = None
        self.C = None

    def fit(self, train_ids, val_ids, data_dir, seed):
        data_dir = Path(data_dir)
        vocab = load_json(data_dir, 'vocab.json')
        token_ids = [int(token_id) for token_id in vocab.values()]
        if not token_ids or min(token_ids) < 0:
            raise ValueError('vocab.json must contain non-negative token ids')
        self.vocab_size = max(token_ids) + 1

        checkpoints = load_json(data_dir, 'checkpoints.json')
        self.checkpoints = [int(checkpoint) for checkpoint in checkpoints]
        if not self.checkpoints or min(self.checkpoints) <= 0:
            raise ValueError('checkpoints.json must contain positive checkpoints')
        self.cap = max(self.checkpoints)

        meta = load_meta(data_dir).set_index('sample_id')
        rng = np.random.default_rng(seed)
        features = []
        labels = []
        label_to_index = {label: index for index, label in enumerate(LABELS)}

        for sample_id in train_ids:
            if sample_id not in meta.index:
                raise ValueError(f'Unknown training sample id: {sample_id}')
            row = meta.loc[sample_id]
            if row['split'] != 'train':
                raise ValueError(f'Training sample {sample_id} is not in the train split')
            label = row['label']
            if label not in label_to_index:
                raise ValueError(f'Unknown training label: {label}')

            tok = self._cap_tokens(load_trace(data_dir, sample_id)['tok'])
            eligible = [
                checkpoint
                for checkpoint in self.checkpoints
                if checkpoint <= len(tok)
            ]
            if not eligible:
                eligible = [len(tok)]
            sampled = rng.choice(eligible, size=self.n_prefixes_per_trace, replace=True)
            for checkpoint in sampled:
                features.append(self._feature(tok, int(checkpoint)))
                labels.append(label_to_index[label])

        missing_labels = set(range(len(LABELS))) - set(labels)
        if missing_labels:
            missing = ', '.join(LABELS[index] for index in sorted(missing_labels))
            raise ValueError(f'Training data is missing classes: {missing}')

        val_features = []
        val_labels = []
        for sample_id in val_ids:
            if sample_id not in meta.index:
                raise ValueError(f'Unknown validation sample id: {sample_id}')
            row = meta.loc[sample_id]
            if row['split'] != 'val':
                raise ValueError(f'Validation sample {sample_id} is not in the val split')
            label = row['label']
            if label not in label_to_index:
                raise ValueError(f'Unknown validation label: {label}')
            tok = self._cap_tokens(load_trace(data_dir, sample_id)['tok'])
            val_features.append(self._feature(tok, len(tok)))
            val_labels.append(label_to_index[label])
        if not val_features:
            raise ValueError('Validation data is required to select C')

        self.classifier = None
        self.C = None
        best_score = -1.0
        for C in self.c_grid:
            classifier = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=C,
            )
            classifier.fit(np.asarray(features), np.asarray(labels))
            predictions = [
                int(np.argmax(self._ordered_probabilities(feature, classifier)))
                for feature in val_features
            ]
            score = f1_score(
                val_labels,
                predictions,
                labels=range(len(LABELS)),
                average='macro',
                zero_division=0,
            )
            if score > best_score:
                best_score = score
                self.classifier = classifier
                self.C = C
        return self

    def score_prefixes(self, tok, checkpoints):
        self._require_fitted()
        tok = self._cap_tokens(tok)
        requested = np.asarray(checkpoints, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested <= 0):
            raise ValueError('checkpoints must be a one-dimensional sequence of positives')

        ends = np.minimum(requested, len(tok))
        order = np.argsort(ends, kind='stable')
        counts = np.zeros(self.vocab_size, dtype=np.int64)
        scores = np.empty(len(requested), dtype=np.float64)
        cursor = 0

        for output_index in order:
            end = int(ends[output_index])
            while cursor < end:
                token_id = int(tok[cursor])
                if token_id < 0 or token_id >= self.vocab_size:
                    raise ValueError(f'Token id {token_id} is outside vocab.json')
                if token_id != 0:
                    counts[token_id] += 1
                cursor += 1
            features = self._normalize_counts(counts)
            probabilities = self._ordered_probabilities(features)
            scores[output_index] = probabilities[LABELS.index('ransomware')]

        return scores

    def final_probs(self, tok):
        self._require_fitted()
        tok = self._cap_tokens(tok)
        return self._ordered_probabilities(self._feature(tok, len(tok)))

    def _feature(self, tok, checkpoint):
        tok = self._cap_tokens(tok)
        end = min(int(checkpoint), len(tok))
        counts = np.zeros(self.vocab_size, dtype=np.int64)
        for token in tok[:end]:
            token_id = int(token)
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f'Token id {token_id} is outside vocab.json')
            if token_id != 0:
                counts[token_id] += 1
        return self._normalize_counts(counts)

    def _normalize_counts(self, counts):
        total = counts.sum()
        if total:
            return counts.astype(np.float64) / total
        return counts.astype(np.float64)

    def _ordered_probabilities(self, features, classifier=None):
        classifier = classifier or self.classifier
        raw = classifier.predict_proba(features.reshape(1, -1))[0]
        ordered = np.zeros(len(LABELS), dtype=np.float64)
        for column, label_index in enumerate(classifier.classes_):
            ordered[int(label_index)] = raw[column]
        return ordered

    def _cap_tokens(self, tok):
        return np.asarray(tok)[:self.cap]

    def _require_fitted(self):
        if self.classifier is None:
            raise RuntimeError('BagOfAPIsDetector must be fitted before scoring')
