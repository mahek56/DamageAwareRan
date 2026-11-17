# R5 Integration and QA
**Mission:** Make everything reproducible and give the team a demo.

**Read:** docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, last 40 lines of docs/STATUS.md.

## Steps
1. tests/: causality helper (modify events after c, the score at c must not change), leakage test (no cluster in two splits, vocab from train only), determinism test, contracts test.
2. scripts/run_all.py: one command from data to table.
3. src/damageaware/cli.py: python -m damageaware.cli report --reports DIR --model gru --alpha 0.05 scores a folder of compatible CAPE reports and writes an HTML or CSV report. Document the required input format. Prototype only.
4. scripts/replay_demo.py: for one trace, plot score vs events for each model with alert markers and the damage counter. This is the expo demo.
5. Review PRs. Keep configs/default.yaml correct.

## Done when
pytest passes. run_all works on stub and real data. Demo image exists.

## Do not
Do not support arbitrary formats. Only the documented CAPE format.

## Handoff
Append a STATUS entry when each step is done. State exact run commands and where outputs are.
