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
