# LFM2.5-230M decision accuracy experiments

This is a small, hand-written *decision-path* experiment, not an independent model benchmark. The production inference path remains pure RSS. The optional Python script here uses official Transformers only to compare readout methods and audit Flint's logits; it is not needed to run the RSS entry point.

## Reproduce

Use the official `LiquidAI/LFM2.5-230M` files (`model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `config.json`) in a local model directory. For the offline comparison install PyTorch and Transformers; this run used CPU PyTorch 2.14.0 and Transformers 5.17.0. From the repository root:

```bash
python3 -m unittest discover -s tests -p test_jev_accuracy_eval.py
python3 scripts/jev_accuracy_eval.py --model /path/to/LFM2.5-230M \
  --cases tests/fixtures/jev_accuracy_development.json --output /tmp/jev-dev.jsonl
python3 scripts/jev_accuracy_eval.py --model /path/to/LFM2.5-230M \
  --cases tests/fixtures/jev_accuracy_holdout.json --output /tmp/jev-holdout.jsonl
```

Each case is scored in its original order, reversed order (the gold label stays attached to its meaning), question paraphrase, and with an unrelated sentence appended to the state. The evaluator loads the model once per fixture, appends every completed case/variant to JSONL, and resumes using case ID plus variant. Use a fresh output path if editing the fixture. `raw` sums option-token conditional log probabilities, `mean` divides by the number of option tokens, `pmi` subtracts the per-token score with state replaced by `N/A`, and `letter` scores a single ` A` or ` B` token under a different answer-marker prompt. The latter is a **prompt change**, not solely a scoring change. `letter_option_mass` measures probability on these two letter tokens before renormalization. Dual-order aggregation adds scores by semantic label before choosing the larger total; it requires both orders. All outputs remain uncalibrated.

Flint RSS defaults to `--scoring token-mean` (option-token average); `--scoring total` selects the prior total-sequence-likelihood behavior. The top-level `scoring` reports the selected mode; each distribution row includes its original `log_likelihood` and actual `ranking_score` used for its probability. Other request fields and `noul`/`choice`/`score` kinds are unchanged. This default is supported only by these small, correlated tests and remains subject to revision on more independent cases.

## Observed results

- On the **28 development questions**, correct original-order choices were `raw 11/28`, `mean 16/28`, `pmi 14/28`, `letter 8/28`. Reversing the answers changed semantic predictions in `12/28`, `8/28`, `6/28`, `19/28` respectively. The original first answer was selected `20/28`, `17/28`, `15/28`, `13/28` respectively.
- On the **20 separate museum questions**, correct original-order choices were `raw 11/20`, `mean 13/20`, `pmi 10/20`, `letter 11/20`. Reversing changed predictions in `8/20`, `7/20`, `8/20`, `12/20`. The four missing-information cases were answered correctly `raw 2/4`, `mean 4/4`, `pmi 2/4`, `letter 2/4`.
- For `mean`, swapping answers changed `8/28` and `7/20` predictions; paraphrases changed `3/28` and `1/20`, and an unrelated sentence changed `1/28` and `0/20`. Dual-order aggregation of `mean` remained **16/28 and 13/20**; extra inference did not raise measured accuracy. This experiment supplies diagnostics, not a production order-ensemble.
- Using the same tokenized prompt/candidate, official Transformers FP32 and Flint raw candidate log-likelihoods differed by at most `0.000044` on the development set. Matching chat-template rendering was also checked using the official tokenizer configuration. The production RSS `token-mean` choices matched the offline evaluator on **all 28 + 20 original-order questions**, giving **16/28 + 13/20** correct.

Most museum questions reuse one state. Several questions within either fixture are correlated, and neither set establishes a general accuracy estimate. In particular, both modes got **0/3 development arithmetic** and **0/2 museum arithmetic** correct in original order. Missing-information success can reflect a preference for the longer `Cannot determine` label under mean scoring; test many more independent contexts and answer phrasings before changing the default.
