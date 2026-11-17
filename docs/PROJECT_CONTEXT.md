# DamageAware: shared context (read this first, 2 minutes)
**Question.** At the same trace-level false-alarm rate, which streaming detector has the fewest file modifications before its first ransomware alert, and does ranking by that damage differ from ranking by fraction of trace seen?

**Not a product.** A research evaluation. We replay recorded sandbox reports. No live monitoring, no prevention claims, no hospital deployment.

**Data.** EN2025 raw CAPE JSON reports only: benign, ransomware, rat, spyware (about 3,180 usable [VERIFY]). Ignore PCAP files (only benign has them) and the CAPE `info` block. Do not use MLRan, MalBehavD or XRan CSVs.

**Models (today).** bag-of-APIs, GRU (unidirectional), XRan-style CNN. Transformer is optional and NOT today.

**Core definitions**
- Score s(c) = P(ransomware | first c events). 4-class softmax, ransomware probability.
- Checkpoints c: 8..128 step 8, then 160..1024 step 32. Same for all models. If a trace has fewer than c events, use min(c, n_events).
- Alert = first checkpoint with s(c) >= tau.
- Trace-level false alarm: a benign trace is flagged if ANY checkpoint score >= tau. FPR = flagged benign / all benign.
- tau per model: smallest tau on VALIDATION benign traces with flagged fraction <= alpha. Alphas: 0.05 (primary), 0.10. Report the realized test FPR with a bootstrap interval. Never tune on test.
- Damage before alert = sum of `dmg` over events up to the alert checkpoint.
- Never-caught ransomware traces get worst-case damage (`total_damage_full`) and are counted in medians. Also report the fraction never caught.
- Damage levels (named honestly): D1 distinct files with successful modification; D2 successful modifying calls; D3 events before alert (no damage claim). R1 decides the level from the audit.
- Split naming: `cluster_held_out` (behavior-clustered groups, primary) and `random` (secondary). Never call it family-held-out unless real family labels are used.
- Train every model on random prefixes so prefix evaluation is fair.

**Report:** realized FPR, recall, median and 90th percentile damage, never-caught fraction, median trace fraction at alert, 4-class macro-F1 on full traces, ms per event, 3 seeds, bootstrap intervals.

**Not doing:** cross-dataset work, dashboard, SHAP, robustness augmentation, live monitoring.
