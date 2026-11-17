# R2 Bag-of-APIs and GRU
**Mission:** Implement two detectors that follow the Detector interface and write score files.

**Read:** docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, last 40 lines of docs/STATUS.md.

## Steps
1. models/bag.py: count vector of the first c events (L1 normalized) with logistic regression, trained on random prefixes of train traces. 4 classes.
2. models/gru.py: embedding 64, one unidirectional GRU layer hidden 128, per-position 4-class head, loss on every position with early events down-weighted, max length 1024, AdamW, early stopping on validation macro-F1.
3. scripts/run_model.py: train with a seed, write scores for val and test at all checkpoints using min(c, n_events).
4. Work on data/processed_stub until R1 posts REAL DATA READY, then change data_dir only.

## Done when
Both models produce score files in the contract format for seeds 0, 1, 2. Causality test passes (R5 provides it).

## Do not
No bidirectional layers. Do not read the test split while training.

## Handoff
Append a STATUS entry when each step is done. State exact run commands and where outputs are.
