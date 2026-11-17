# R3B Causal Transformer
**Mission:** Implement the causal Transformer (architecture figure supplied by lead) as a prefix-safe detector. Same interface, same training rules as the GRU, so it is a fair fourth model.

**Read:** docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, last 40 lines of docs/STATUS.md. Also read src/damageaware/models/gru.py (R2) and scripts/run_model.py for the recipe and runner pattern to match.

## Architecture spec (from the lead's figure, adapted to our data contract)
| Figure block | Our implementation |
|---|---|
| 1 Behavioral input, 2 Feature extraction | NOT built. Input is the token id sequence `tok` from the data contract (R1 already did extraction). No hand-made numeric features, no trace length or metadata. |
| 3 Feature embedding | `nn.Embedding(vocab_size, 128, padding_idx=0)`, vocab_size = max(id) + 1 (same bug-fix as bag). Scale by sqrt(128). |
| 4 Positional encoding | Sinusoidal, added to embeddings, max length from checkpoints.json (1024), not hard-coded. |
| 5 Causal decoder-only stack | N blocks (default N=2, allowed 2 to 4). Each block: causal multi-head self-attention (H=4 heads, d=128) + residual + LayerNorm, then FFN Linear(d, 4d) - GELU - Linear(4d, d) + residual + LayerNorm (post-LN as in the figure). Dropout 0.1. Built from standard PyTorch layers with an upper-triangular causal mask. |
| 6, 7 Final representation, head | `Linear(128, 4)` applied at EVERY position, softmax over 4 classes (benign, ransomware, rat, spyware). |
| 8 Output | Score = P(ransomware) = softmax index 1. Not a 2-class output (the figure shows 2 classes, our decision D08 says 4). |

Right padding plus causal mask keeps real positions from ever seeing PAD or the future. Handle the empty trace without NaN (return zeros, as bag and GRU do).

## Steps
1. models/transformer.py with the spec above. Hyperparameters (layers, heads, width, dropout) are constructor defaults, not runner flags.
2. Training, same as GRU except as noted: per-position cross-entropy on non-PAD positions, w_t = min(1, t/64), class weights from train frequencies only, AdamW (weight decay 1e-2), batch 32, gradient clipping 1.0, up to 30 epochs, early stopping on full-trace validation macro-F1 (patience 4, restore best). ONE deliberate difference: lr 5e-4 with short linear warmup (about 200 steps) because post-LN Transformers are less stable at lr 1e-3. Document it. Seed Python, NumPy, PyTorch.
3. Scoring: one forward pass per trace, read softmax at position min(c, len) - 1 for each checkpoint.
4. Register `transformer` in scripts/run_model.py (ONE registry line, see CR-R3B). Reuse --max-epochs.
5. tests/test_models_r3b.py: shapes, causality (change events after c, score at c identical), PAD never attended (changing the amount of right padding leaves scores unchanged), smoke training on stub, short and empty traces, vocab with a gap, runner output for `--model transformer`.
6. scripts/latency.py: ms per new event for the Transformer with and without KV-cache (recompute full prefix vs cache keys/values). Engineering number only.
7. docs/TRANSFORMER_NOTES.md: every design choice, every deviation from the figure, parameter count next to the GRU's.

## Done when
Score files for seeds 0, 1, 2 on stub, then real data after R1 posts REAL DATA READY. All tests pass. Parameter count reported.

## Do not
No bidirectional or non-causal attention. Do not read the test split while training. No metadata features. Do not claim it beats the CNN or GRU.

## Handoff
Append a STATUS entry when each step is done. State exact run commands and where outputs are.
