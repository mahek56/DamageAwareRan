# Interfaces (the contract between roles)
Processed data directory `data/processed` (stub: `data/processed_stub`). Same format for both. Loaders: `src/damageaware/contracts.py`.

**meta.csv** columns: sample_id, label (benign|ransomware|rat|spyware), split (train|val|test, primary cluster_held_out), split_random, cluster_id, n_events (events seen by models, <= 1024), total_damage_full (damage over the FULL trace).
**traces/{sample_id}.npz**: `tok` int32 [n_events] token ids; `dmg` int32 [n_events] damage added by each event (unit = damage level, see PROJECT_CONTEXT). Order = event order.
**vocab.json**: token name -> id. **checkpoints.json**: list of ints (D04).
**Scores** `results/scores/{model}_{split}_seed{k}.npz`: `sample_ids` [n], `checkpoints` [m], `scores` float32 [n, m] = P(ransomware | first min(c, n_events) events), optional `final_probs` float32 [n, 4] (class order benign, ransomware, rat, spyware) for macro-F1.
**Detector interface** (src/damageaware/models/base.py): `fit(train_ids, val_ids, data_dir, seed)`, `score_prefixes(tok, checkpoints) -> np.ndarray[m]`, `final_probs(tok) -> np.ndarray[4]`. score_prefixes at c may use only tok[:c].
**Runner**: `python scripts/run_model.py --model {bag,gru,cnn} --data-dir DIR --seed K` writes score files for val and test.
**Evaluator**: `python scripts/evaluate.py --data-dir DIR --models bag gru cnn --alphas 0.05 0.10` writes `results/tables/main.csv` and `results/figures/*.png`.
**Config**: `configs/default.yaml` key `data_dir` selects stub or real data. Switching to real data is one line.
Ownership and changes: see docs/ROLES.md. Interfaces change only via CHANGE REQUEST.
