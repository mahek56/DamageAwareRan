# Frozen decisions (change only through a CHANGE REQUEST approved by the lead)
| ID | Decision |
|---|---|
| D01 | Dataset: EN2025 raw reports only |
| D02 | Alphas 0.05 (primary) and 0.10 |
| D03 | Alert policy: first checkpoint with score >= tau (k=1). k=3 as sensitivity check later |
| D04 | Checkpoints: 8..128 step 8, 160..1024 step 32 |
| D05 | N_CAP = 1024 events seen by models. Damage totals come from the full trace |
| D06 | Splits: cluster_held_out 70/15/15 primary, random secondary. Dedup before splitting |
| D07 | Vocabulary built from train split only. 0=PAD, 1=UNK |
| D08 | 4 classes in training, score = P(ransomware) |
| D09 | Seeds: 0,1,2 |
| D10 | Damaging-API list frozen by R1 in configs/damaging_apis.yaml after the audit, before any model results |
| D11 | Never-caught traces get worst-case damage |
| D12 | GRU unidirectional. All models trained on random prefixes |
| D13 | Benign FPR is primary. FPR over all non-ransomware is secondary |
