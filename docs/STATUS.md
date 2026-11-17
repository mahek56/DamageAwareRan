# STATUS (append-only handoff log; LLMs read only the last 40 lines)
**Current gate:** G0 setup. Next gate: G-REAL (R1 delivers real data).

## Handoff template (copy, fill, append at the bottom)
```
### [date time] R? task name
Done: what exists now
Files: paths changed
Run: exact command to reproduce
Output: where results are
Open: problems, assumptions, things others must know
Needs from: R? (what)
```

## Change requests (lead approves, then updates DECISIONS or INTERFACES)
(none yet)

## Handoff log
### [2026-11-17 14:05 +05:30] R2 Bag-of-APIs A2
Done: Bag detector selects C from [1, 10, 100, 1000] by validation macro-F1 and handles vocab gaps, short traces, and empty traces.
Files: src/damageaware/models/bag.py, tests/test_models_r2.py, docs/STATUS.md
Run: `$env:PYTHONPATH = 'src'; .\.venv\Scripts\python.exe -m pytest -q`
Output: pytest stdout; 10 passed. No model score files were produced.
Open: GRU not started.
Needs from: none

### [2026-11-17 15:00 +05:30] R2 GRU detector
Done: Added a CPU-only unidirectional GRU with masked position-weighted loss, train-only class weights, validation macro-F1 early stopping, and causal prefix scoring.
Files: src/damageaware/models/gru.py, scripts/run_model.py, tests/test_models_r2.py, docs/STATUS.md
Run: `$env:PYTHONPATH = 'src'; foreach ($seed in 0, 1, 2) { .\.venv\Scripts\python.exe scripts\run_model.py --model gru --data-dir data\processed_stub --seed $seed --out-dir results\scores }`
Test: `$env:PYTHONPATH = 'src'; .\.venv\Scripts\python.exe -m pytest -q`
Output: results/scores/gru_{val,test}_seed{0,1,2}.npz; IDs, checkpoints, and score shapes load-checked. Pytest: 18 passed.
Open: none.
Needs from: none

### [2026-11-17 14:23 +05:30] R2 Bag runner
Done: Added a lazy bag-only runner, validation macro-F1 reporting, and score-file contract tests.
Files: scripts/run_model.py, tests/test_models_r2.py, docs/STATUS.md
Run: `$env:PYTHONPATH = 'src'; foreach ($seed in 0, 1, 2) { .\.venv\Scripts\python.exe scripts\run_model.py --model bag --data-dir data\processed_stub --seed $seed --out-dir results\scores }`
Test: `$env:PYTHONPATH = 'src'; .\.venv\Scripts\python.exe -m pytest -q`
Output: results/scores/bag_{val,test}_seed{0,1,2}.npz; load checks passed for IDs, checkpoints, and array shapes. Pytest: 12 passed.
Open: GRU remains unstarted.
Needs from: none

### [2026-11-17 19:50 +05:30] R5 integration QA
Done: Fixed the pytest bootstrap for bare `pytest -q`, added the causal/leakage/determinism/round-trip QA checks, and validated the contract behavior for capped-damage totals and score serialization.
Files: tests/conftest.py, tests/helpers.py, tests/test_contracts.py, tests/test_qa.py, src/damageaware/contracts.py, docs/STATUS.md
Run: `python -m pytest -q` and `pytest -q`
Output: both commands pass in the repo root; 22 passed, 1 skipped, 1 warning. Stub data generated at data/processed_stub and score checks passed without reading the test split for tuning.
Open: No blockers. `scripts/run_all.py`, `scripts/replay_demo.py`, and `src/damageaware/cli.py` were not started per role instructions.
Needs from: none

### [2026-11-17 21:23 +05:30] R5 replay demo S2
Done: Added a single-trace replay plot with sample-id-based score lookup, checkpoint validation, optional validation-derived tau lines and alert markers, and cumulative damage. Missing evaluation thresholds are reported and plots continue without tau.
Files: scripts/replay_demo.py, tests/test_demo_r5.py, docs/STATUS.md
Run: `$env:PYTHONPATH = 'src'; python scripts/replay_demo.py --data-dir data/processed_stub --models bag gru --seed 0 --out results/figures/replay_stub.png --tau-from-val`; `pytest -q`
Output: results/figures/replay_stub.png (114524 bytes); pytest: 24 passed, 1 skipped, 1 warning.
Open: Stub plot is labeled STUB DATA. `src/damageaware/eval/thresholds.py` is not present, so tau plotting reports that thresholds are unavailable.
Needs from: none
