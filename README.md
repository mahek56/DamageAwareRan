# DamageAware team kit

Question: at the same false-alarm rate, how many file modifications have happened when each detector first alerts?

## Setup (lead, once)
1. Create the git repo and copy this kit into it. Put EN2025 raw reports in `data/raw/` (read-only, not in git).
2. `pip install -r requirements.txt`
3. `PYTHONPATH=src python scripts/make_stub_data.py --scores` then `PYTHONPATH=src pytest -q`. Must pass.
4. Paste `.agent/rules/damageaware.md` into Antigravity rules (and keep AGENTS.md at repo root). Check the exact rules location in Antigravity docs.
5. Each member creates a branch named after their role, e.g. `r1-data`.

## How each member starts (every new chat)
Paste the start prompt in `docs/LLM_START_PROMPT.md`, with the role number filled in. It points the LLM at four small files. Do not paste anything else.

## Order of work
See `docs/ROLES.md`. Short version: R1 builds real data. R2 to R6 start right now on stub data in the same format. When R1 delivers real data, everyone switches one config line.

## Rules in one breath
Own only your files. Never edit docs/DECISIONS.md or docs/INTERFACES.md yourself, raise a change request in docs/STATUS.md. Finish every task by writing a handoff entry in docs/STATUS.md and committing.
