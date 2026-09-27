# Pure-RSS typed decisions on Flint

`scripts/jev_lfm2_5.rss` runs LiquidAI/LFM2.5-230M text inference and Jev-inspired typed decisions inside Flint. Model forward, candidate scoring, normalization and JSON serialization are RSS. It uses **only existing Flint host functions**; no Python inference process or new host API is required.

## Inputs and output

The `--request` argument is JSON with a shared `state` and an ordered `questions` array. Each question has a unique application-defined `id`, a `kind`, a `prompt`, and a `choices` array:

- `noul`: empty `choices`; answers `No` and `Yes` (the `Yes` probability is the positive outcome).
- `choice`: 2–16 distinct, nonempty strings in `choices`.
- `score`: empty `choices`; answers `0` through `5`. These are *choice labels*, not calibrated numeric ratings.

The output contains ordered `results`; each has `id`, `kind`, `selected`, `distribution` entries (`label`, `probability`, `log_likelihood`) and `calibrated:false`. Top-level fields identify the model and the supplied temperature. Each question is evaluated against the same shared state; candidate and question caches are cleared independently. `id` never enters the model prompt.

For every candidate, the script tokenizes the fixed chat prefix plus a leading-space candidate and checks that the prefix token IDs match; mismatch fails instead of reporting a misaligned likelihood. It teacher-forces each candidate token and sums its full-vocabulary conditional log probabilities. By default (`--scoring token-mean`) it divides each candidate's log-likelihood sum by that candidate's token count before normalizing scores with a maximum-shifted softmax. `--scoring total` retains the original total-sequence-likelihood ranking. No answer text is sampled/generated.

Flint's current host does not expose scalar tensor readback: the RSS uses `argmax_int` comparisons to recover token IDs and approximates each `log(probability)` with 24 binary-search rounds on `[0, 80]`. This has less than 0.000005 resolution in log units. A token probability below `exp(-80)` fails explicitly. Prompt and candidate token caps are enforced; there is no silent truncation. All probabilities are **uncalibrated**. Do not interpret `selected`, the `score` option, or a high probability as validated correctness; fitting/evaluating calibration needs a separately labeled dataset.

## Run

Download `model.safetensors` and `tokenizer.json` from [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M/tree/main) into a local model directory. The model has a separate [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-230M/blob/main/LICENSE); weights are not kept in this repository. From the Flint repository root, with a compatible `flint` executable and its Torch runtime available:

```bash
flint --llama --device cpu --script scripts/jev_lfm2_5.rss -- \
  --model /path/to/LFM2.5-230M/model.safetensors \
  --tokenizer /path/to/LFM2.5-230M/tokenizer.json \
  --request '{"state":"A book is on the desk.","questions":[{"id":"q1","kind":"noul","prompt":"Is the book on the desk?","choices":[]},{"id":"q2","kind":"choice","prompt":"Where is it?","choices":["on the desk","in the garden"]},{"id":"q3","kind":"score","prompt":"How certain?","choices":[]}]}'
```

Use `--temperature 1.0` by default; another positive value rescales only the candidate distribution and does not establish calibration. The default `--scoring token-mean` ranks by each candidate's average per-token log probability; `--scoring total` restores the earlier total-sequence-likelihood ranking. The response reports `scoring` and each entry's `ranking_score`; `log_likelihood` always retains the original summed log probability. See the [fixed-case accuracy and robustness experiments](jev-accuracy-experiments.md) for measured benefits and limitations. The runner rejects unsupported kinds, duplicate choices, overlong candidates/prompts, invalid scoring and nonpositive temperature. `assert` failures exit without a partial JSON result.

For a model-free RSS math check, run `flint --llama --script scripts/jev_math_test.rss`. For the existing-host readback check, run `flint --llama --device cpu --script scripts/jev_host_readback_test.rss -- --tokenizer /path/to/tokenizer.json`. The latter only needs a tokenizer and the Torch runtime.

This is an open-weight zero-shot imitation of Jev's typed decision *interface*; it is not TypeSafe's model, a trained Jev replica, or a calibrated benchmark result. The prototype repeats the model prefill for each option.
