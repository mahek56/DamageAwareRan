# R1 Data and audit (critical path)
**Mission:** Produce the real dataset in the contract format and decide the damage level.

**Read:** docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, last 40 lines of docs/STATUS.md.

## Steps
1. Print keys of 3 raw reports and one call. Check fields: api name, order or timestamp, arguments, status, repeated. Use real files only.
2. Write scripts/audit.py. Output docs/AUDIT.md: class counts, trace length percentiles, timestamp validity, names of write/rename/delete APIs, whether file paths or handles are resolvable, distinct files per ransomware trace, dud runs, duplicate sequences, a shortcut probe (classifier on trace length and process count only; report AUC), and which CAPE info fields differ by class.
3. Decide damage level D1, D2 or D3 and freeze configs/damaging_apis.yaml. Post it in STATUS and wait for lead approval.
4. parse.py: order events by timestamp with a stable sort, tie rule process id then original index, fallback per-process order if timestamps are bad. Collapse repeated calls but keep original counts in dmg. Cap at 1024 events and keep full-trace total damage in meta.
5. splits.py: remove exact duplicate sequences, cluster near-duplicates (MinHash on API n-grams), build cluster_held_out 70/15/15 stratified by label, plus a random split.
6. scripts/build_data.py: one command builds data/processed with meta.csv, traces/, vocab.json (train only), checkpoints.json.
7. FIRST deliver a mini real dataset of about 50 traces so others can test shapes. Then the full one.

## Done when
pytest tests/test_contracts.py passes on data/processed. AUDIT.md written. One command rebuilds.

## Do not
Do not call anything an encrypted file. Do not use PCAP.

## Handoff
Append a STATUS entry when each step is done. State exact run commands and where outputs are.
