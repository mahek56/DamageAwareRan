# DamageAware project rules
1. Never modify anything in data/raw. It is read-only.
2. Training and tuning code may use only train and val splits. Only scripts/evaluate.py reads the test split.
3. Fit vocabulary, class weights and scalers on the train split only.
4. Every random seed comes from configs/default.yaml and is logged.
5. Save every results table with the config used and the git commit hash.
6. Ask before any command that deletes files, installs packages, or runs over 10 minutes.
7. Do not change docs/DECISIONS.md or docs/INTERFACES.md. If a change seems needed, stop and write a CHANGE REQUEST in docs/STATUS.md.
8. Only edit files your role owns (docs/ROLES.md). Ask first for anything else.
9. Write a test before a long training run. Run pytest before saying a task is done.
10. Report numbers only from files the code produced. Never estimate from memory. If a number cannot be computed, say so.
11. Field names in the report JSON must come from real files. Print example records first.
12. Models must be causal in the prefix sense: the score at checkpoint c may use only the first c events. Never use bidirectional layers or future events.
13. Do not call an API call an "encrypted file". Use the damage level names in docs/PROJECT_CONTEXT.md.
14. Keep context small: read only docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, your role file and the last 40 lines of docs/STATUS.md. Do not paste whole files or long logs into chat, only the last 30 lines of an error.
15. End every task by appending a handoff entry to docs/STATUS.md and committing.
