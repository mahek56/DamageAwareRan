# Roles, ownership, execution order
| Role | Mission | Owns (only edit these) | Needs | Can start now? |
|---|---|---|---|---|
| R1 Data and audit | Audit EN2025, damage level, parser, splits, real data | src/damageaware/parse.py, splits.py, scripts/audit.py, scripts/build_data.py, configs/damaging_apis.yaml, docs/AUDIT.md | raw data | Yes. Critical path |
| R2 Bag and GRU | bag-of-APIs and GRU detectors | src/damageaware/models/bag.py, gru.py, scripts/run_model.py, tests/test_models_r2.py | stub, then real data | Yes, on stub |
| R3 CNN | XRan-style CNN | src/damageaware/models/cnn.py, docs/XRAN_NOTES.md | stub, then real data | Yes, on stub |
| R4 Evaluation | thresholds, damage, metrics, tables, figures | src/damageaware/eval/*, scripts/evaluate.py, tests/test_eval_r4.py | stub scores, then real | Yes, on stub scores |
| R5 Integration and QA | tests, run_all, CLI report, replay demo | tests/*, scripts/run_all.py, scripts/replay_demo.py, src/damageaware/cli.py, configs/ | everything | Yes |
| R6 Research and writing | literature, novelty log, paper skeleton, Hancock case | docs/paper/* | none | Yes |
Lead (you): approves change requests, edits DECISIONS and INTERFACES, merges to main.

## Execution order
```
T0  Setup (all, 30 min): clone, install, run stub, pytest passes. Checkpoint C1.
T1  Parallel:
    R1  real audit -> mini real dataset (about 50 traces, in contract format) -> full data
    R2  bag, GRU on stub
    R3  CNN on stub
    R4  evaluator on stub scores
    R5  tests, run_all, CLI skeleton on stub
    R6  literature, search log, paper skeleton
T2  GATE G-REAL: R1 posts "REAL DATA READY" in STATUS. Checkpoint C2.
    R2, R3 set data_dir to real, train. R4 and R5 rerun on real.
T3  R2, R3 write score files. R4 evaluates. R5 runs run_all end to end. Checkpoint C3.
T4  First real table reviewed by lead and all. Then 3 seeds, bootstrap, extras. Checkpoint C4.
```
Dependency rule: only R1's real data blocks others, and the stub removes that block. If the audit changes the damage level, only the meaning of `dmg` changes, not the file format.

## Coordination
- Interfaces are frozen. Disagree? Write a CHANGE REQUEST in STATUS, do not work around it.
- One branch per role. PRs into main are reviewed by R5, merged by lead.
- Checkpoints C1 to C4: lead calls them, each role confirms in STATUS in one line.
- Blockers go into STATUS with "Needs from: R?".
