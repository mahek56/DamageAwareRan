"""Causal, unidirectional GRU detector."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from damageaware.contracts import LABELS, load_json, load_meta, load_trace
from damageaware.models.base import Detector


class _PositionGRU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.classifier = nn.Linear(128, len(LABELS))

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        hidden, _ = self.gru(embedded)
        return self.classifier(hidden)


class GRUDetector(Detector):
    name = 'gru'
    batch_size = 32
    patience = 4

    def __init__(self, max_epochs=30):
        if not 1 <= max_epochs <= 30:
            raise ValueError('max_epochs must be between 1 and 30')
        self.max_epochs = max_epochs
        self.model = None
        self.checkpoints = None
        self.cap = None
        self.vocab_size = None
        self.best_epoch = None

    def fit(self, train_ids, val_ids, data_dir, seed):
        torch.set_num_threads(os.cpu_count() or 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        label_to_index = {label: index for index, label in enumerate(LABELS)}
        train_sequences, train_labels = self._load_split(
            train_ids, 'train', meta, label_to_index, data_dir
        )
        val_sequences, val_labels = self._load_split(
            val_ids, 'val', meta, label_to_index, data_dir
        )
        if not train_sequences:
            raise ValueError('Training data is required')
        if not val_sequences:
            raise ValueError('Validation data is required for early stopping')

        class_counts = np.bincount(train_labels, minlength=len(LABELS))
        if np.any(class_counts == 0):
            missing = ', '.join(
                LABELS[index]
                for index, count in enumerate(class_counts)
                if count == 0
            )
            raise ValueError(f'Training data is missing classes: {missing}')
        class_weights = len(train_labels) / (len(LABELS) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.model = _PositionGRU(self.vocab_size).to('cpu')
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )
        best_state = None
        best_score = -1.0
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            order = torch.randperm(len(train_sequences)).tolist()
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start:start + self.batch_size]
                sequences = [train_sequences[index] for index in batch_indices]
                labels = [train_labels[index] for index in batch_indices]
                if not any(len(sequence) for sequence in sequences):
                    continue
                token_batch, _ = self._collate(sequences)
                logits = self.model(token_batch)
                max_length = token_batch.shape[1]
                targets = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
                targets = targets.expand(-1, max_length)
                losses = F.cross_entropy(
                    logits.reshape(-1, len(LABELS)),
                    targets.reshape(-1),
                    weight=class_weights,
                    reduction='none',
                ).reshape(len(sequences), max_length)
                mask = token_batch.ne(0)
                position_weights = (
                    torch.arange(1, max_length + 1, dtype=torch.float32)
                    .div(64)
                    .clamp(max=1)
                    .unsqueeze(0)
                )
                weighted_mask = position_weights * mask
                denominator = weighted_mask.sum()
                if denominator.item() == 0:
                    continue
                loss = (losses * weighted_mask).sum() / denominator

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            val_probs = self._final_probs_for_sequences(val_sequences)
            val_predictions = np.argmax(val_probs, axis=1)
            score = f1_score(
                val_labels,
                val_predictions,
                labels=range(len(LABELS)),
                average='macro',
                zero_division=0,
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError('GRU training did not produce a validation checkpoint')
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def score_prefixes(self, tok, checkpoints):
        self._require_fitted()
        tok = self._cap_tokens(tok)
        requested = np.asarray(checkpoints, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested <= 0):
            raise ValueError('checkpoints must be a one-dimensional sequence of positives')
        scores = np.zeros(len(requested), dtype=np.float32)
        if not len(tok) or not len(requested):
            return scores

        positions = np.minimum(requested, len(tok)) - 1
        token_batch = torch.as_tensor(tok, dtype=torch.long).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(token_batch)[0]
            selected = logits[torch.as_tensor(positions, dtype=torch.long)]
            probabilities = torch.softmax(selected, dim=-1)
        return probabilities[:, LABELS.index('ransomware')].cpu().numpy()

    def final_probs(self, tok):
        self._require_fitted()
        tok = self._cap_tokens(tok)
        if not len(tok):
            return np.zeros(len(LABELS), dtype=np.float32)

        token_batch = torch.as_tensor(tok, dtype=torch.long).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(token_batch)
            probabilities = torch.softmax(logits[0, -1], dim=-1)
        return probabilities.cpu().numpy()

    def _load_split(self, sample_ids, split, meta, label_to_index, data_dir):
        sequences = []
        labels = []
        for sample_id in sample_ids:
            if sample_id not in meta.index:
                raise ValueError(f'Unknown {split} sample id: {sample_id}')
            row = meta.loc[sample_id]
            if row['split'] != split:
                raise ValueError(f'Sample {sample_id} is not in the {split} split')
            label = row['label']
            if label not in label_to_index:
                raise ValueError(f'Unknown {split} label: {label}')
            tok = load_trace(data_dir, sample_id)['tok']
            sequences.append(self._cap_tokens(tok))
            labels.append(label_to_index[label])
        return sequences, labels

    def _cap_tokens(self, tok):
        tokens = np.asarray(tok)
        if tokens.ndim != 1:
            raise ValueError('tok must be one-dimensional')
        tokens = tokens[:self.cap]
        if tokens.size and (np.any(tokens < 0) or np.any(tokens >= self.vocab_size)):
            raise ValueError('Token id is outside vocab.json')
        return tokens

    def _collate(self, sequences):
        lengths = [len(sequence) for sequence in sequences]
        token_batch = torch.zeros(
            (len(sequences), max(lengths)),
            dtype=torch.long,
        )
        for row, sequence in enumerate(sequences):
            if lengths[row]:
                token_batch[row, :lengths[row]] = torch.as_tensor(
                    sequence,
                    dtype=torch.long,
                )
        return token_batch, lengths

    def _final_probs_for_sequences(self, sequences):
        probabilities = np.zeros((len(sequences), len(LABELS)), dtype=np.float32)
        nonempty = [index for index, sequence in enumerate(sequences) if len(sequence)]
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(nonempty), self.batch_size):
                batch_indices = nonempty[start:start + self.batch_size]
                batch_sequences = [sequences[index] for index in batch_indices]
                token_batch, lengths = self._collate(batch_sequences)
                logits = self.model(token_batch)
                batch_probs = torch.softmax(logits, dim=-1)
                for row, (index, length) in enumerate(zip(batch_indices, lengths)):
                    probabilities[index] = batch_probs[row, length - 1].cpu().numpy()
        return probabilities

    def _require_fitted(self):
        if self.model is None:
            raise RuntimeError('GRUDetector must be fitted before scoring')
