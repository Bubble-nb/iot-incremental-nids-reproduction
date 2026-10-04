# Completed fixed-sample experiments

These summaries come from the completed forward runs (seeds 1, 2, 3) and one reverse run (seed 1), each allowing up to 200 epochs with validation-based model selection. They are not the one-epoch execution demonstrations.

- repeated_run_metrics.csv contains 48 independently recomputed domain results for 24 model states.
- forward_repeats.md reports every paired difference and descriptive means. Combined improvements did not consistently repeat.
- reverse_single_run.md reports both classification performance and false-positive costs.
- repeat_validation_audit.json confirms matching forward test feature/semantic-label multisets and differing training-internal validation multisets.

The run seed changes class order, training-internal validation sampling and training randomness, while prep1 train/test membership remains fixed. Thus, this is not an experiment on three independent test splits. Different migration directions are not pooled into one performance estimate.

Macro-F1 averages over classes occurring in the evaluated domain, and predictions outside that domain remain errors. False-positive rate uses truly benign samples as its denominator. Full weights and per-sample predictions are retained locally, not uploaded here. No significance or deployment-superiority claim is made.
