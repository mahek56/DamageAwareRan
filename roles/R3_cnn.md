# R3 XRan-style CNN
**Mission:** Implement the base-paper style CNN as a prefix-safe detector.

**Read:** docs/PROJECT_CONTEXT.md, docs/INTERFACES.md, last 40 lines of docs/STATUS.md.

## Steps
1. models/cnn.py: embedding, 2 convolution layers, global max pooling, dense layer, 4 classes. Input is only the first c events padded with PAD to 1024.
2. Train on random prefix lengths so evaluation on prefixes is fair.
3. Score each checkpoint with a separate cropped forward pass, batched.
4. docs/XRAN_NOTES.md: list every detail taken from the paper and every detail guessed. Read the XRan method section, not only the abstract.
5. Use scripts/run_model.py with --model cnn. Stub first, real data later.

## Done when
Score files for seeds 0, 1, 2. Truncation test passes: changing events after c does not change the score at c.

## Do not
Do not claim this reproduces XRan. It is XRan-style.

## Handoff
Append a STATUS entry when each step is done. State exact run commands and where outputs are.
