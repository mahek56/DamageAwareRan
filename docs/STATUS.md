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
