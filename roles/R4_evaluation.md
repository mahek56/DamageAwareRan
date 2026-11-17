# R4 Evaluation engine
**Mission:** Turn score files into the damage table and figures.

**Read:** docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, last 40 lines of docs/STATUS.md.

## Steps
1. eval/thresholds.py: per model, tau = smallest value on val benign traces such that the fraction with max checkpoint score >= tau is <= alpha.
2. eval/damage.py: alert checkpoint, damage at alert from cumulative dmg, never-caught gets total_damage_full, trace fraction at alert.
3. eval/metrics.py: realized test FPR, recall, median and 90th percentile damage, fraction zero damage, fraction never caught, bootstrap intervals, 4-class macro-F1 from final_probs, ranking comparison (damage vs trace fraction).
4. eval/figures.py: damage CDF per model, ranking chart. scripts/evaluate.py writes results/tables/main.csv and results/figures.
5. Develop on stub scores (python scripts/make_stub_data.py --scores). Add a tiny hand-computed unit test.

## Done when
evaluate.py runs on stub and on real scores. Every number is saved with config and commit hash.

## Do not
Do not tune tau on test. Do not drop never-caught traces.

## Handoff
Append a STATUS entry when each step is done. State exact run commands and where outputs are.
