"""Offline accuracy experiments for Flint's pure-RSS decision scorer.

This evaluator is optional research tooling; the production inference path stays RSS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_cases(cases: list[dict]) -> None:
    ids: set[str] = set()
    for case in cases:
        if (not case.get("id") or case["id"] in ids or not case.get("state")
                or not case.get("prompt") or not case.get("paraphrase")
                or len(case.get("choices", [])) != 2 or len(set(case["choices"])) != 2
                or case.get("gold") not in case["choices"] or not case.get("tag")):
            raise ValueError(f"invalid or duplicate case: {case.get('id')}")
        ids.add(case["id"])


def render_prefix(case: dict, mode: str) -> str:
    choices = case["choices"]
    if mode == "raw":
        listed = "".join("- " + label + "\n" for label in choices)
        assistant = ""
    elif mode == "letter":
        listed = "".join(chr(65 + i) + ". " + label + "\n" for i, label in enumerate(choices))
        assistant = "Answer:"
    else:
        raise ValueError(f"unknown prompt mode: {mode}")
    user_prompt = ("Shared context:\n" + case["state"] + "\n\nQuestion:\n" +
                   case["prompt"] + "\n\nAllowed answers:\n" + listed)
    system = case.get("system", "Choose exactly one listed answer. Do not explain.")
    return ("<|startoftext|><|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n" + assistant)


def continuation_ids(tokenizer, prefix: str, suffix: str) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
    if not prefix_ids or full_ids[:len(prefix_ids)] != prefix_ids or len(full_ids) <= len(prefix_ids):
        raise ValueError("candidate does not extend the exact token prefix")
    return prefix_ids, full_ids[len(prefix_ids):]


def score_methods(answers: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {name: {} for name in ("raw", "mean", "pmi", "letter")}
    for answer in answers:
        label = answer["label"]
        parts = answer["token_logprobs"]
        assert parts
        total = sum(parts)
        result["raw"][label] = total
        result["mean"][label] = total / len(parts)
        result["pmi"][label] = total / len(parts) - answer["null_logprob"] / answer["null_tokens"]
        result["letter"][label] = answer["letter_logprob"]
    return result


def aggregate(scores: dict[str, float]) -> str:
    return max(scores, key=scores.__getitem__)


def make_variants(case: dict, distractor: str) -> list[dict]:
    original = {**case, "name": "original"}
    return [
        original,
        {**case, "name": "reversed", "choices": list(reversed(case["choices"]))},
        {**case, "name": "paraphrased", "prompt": case["paraphrase"]},
        {**case, "name": "distractor", "state": case["state"] + "\n" + distractor},
    ]


def aggregate_both_orders(pair: dict[str, dict], method: str) -> str:
    if set(pair) != {"original", "reversed"}:
        raise ValueError("both orders are required")
    a, b = pair["original"], pair["reversed"]
    if set(a["choices"]) != set(b["choices"]):
        raise ValueError("different semantic choices")
    combined = {label: a["scores"][method][label] + b["scores"][method][label]
                for label in a["choices"]}
    return aggregate(combined)


def paired_summary(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        for method, prediction in row["predictions"].items():
            target = result.setdefault(method, {"correct": 0, "total": 0, "by_tag": {}})
            correct = int(prediction == row["gold"])
            target["correct"] += correct
            target["total"] += 1
            group = target["by_tag"].setdefault(row["tag"], {"correct": 0, "total": 0})
            group["correct"] += correct
            group["total"] += 1
    return result


def read_option_logprobs(model, tokenizer, prefix: str, label: str, torch) -> list[float]:
    prefix_ids, suffix_ids = continuation_ids(tokenizer, prefix, " " + label)
    full_ids = prefix_ids + suffix_ids
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([full_ids[:-1]])).logits[0]
        positions = torch.arange(len(prefix_ids) - 1, len(full_ids) - 1)
        selected = torch.log_softmax(logits[positions].float(), dim=-1).gather(
            -1, torch.tensor(suffix_ids).unsqueeze(-1)
        )
    return selected.flatten().tolist()


def read_letter_logprobs(model, tokenizer, case: dict, torch) -> tuple[dict[str, float], float]:
    prefix = render_prefix(case, "letter")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = []
    for i in range(len(case["choices"])):
        baseline, suffix = continuation_ids(tokenizer, prefix, " " + chr(65 + i))
        if baseline != prefix_ids or len(suffix) != 1:
            raise ValueError("letter readout must be a single token after the same prefix")
        ids.append(suffix[0])
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([prefix_ids])).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)[ids].tolist()
        mass = torch.exp(torch.logsumexp(logits[ids], dim=0) - torch.logsumexp(logits, dim=0)).item()
    return dict(zip(case["choices"], logprobs)), mass


def evaluate_variant(model, tokenizer, case: dict, null_cache: dict, torch) -> dict:
    prefix = render_prefix(case, "raw")
    null_key = (case["prompt"], tuple(case["choices"]))
    if null_key not in null_cache:
        null_prefix = render_prefix({**case, "state": "N/A"}, "raw")
        null_cache[null_key] = {
            label: read_option_logprobs(model, tokenizer, null_prefix, label, torch)
            for label in case["choices"]
        }
    letter, mass = read_letter_logprobs(model, tokenizer, case, torch)
    answers = []
    for label in case["choices"]:
        parts = read_option_logprobs(model, tokenizer, prefix, label, torch)
        prior = null_cache[null_key][label]
        answers.append({"label": label, "token_logprobs": parts, "null_logprob": sum(prior),
                        "null_tokens": len(prior), "letter_logprob": letter[label]})
    scores = score_methods(answers)
    return {"id": case["id"], "variant": case["name"], "tag": case["tag"],
            "gold": case["gold"], "choices": case["choices"], "predictions": {
                name: aggregate(values) for name, values in scores.items()},
            "scores": scores, "answers": answers, "letter_option_mass": mass}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory with tokenizer and config")
    parser.add_argument("--cases", required=True, help="Frozen case fixture")
    parser.add_argument("--output", required=True, help="Append-only JSONL output; resume by ID and variant")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test first N cases when positive")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    validate_cases(cases)
    if args.limit:
        cases = cases[:args.limit]
    output = Path(args.output)
    if output.exists():
        previous = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        done = {(row["id"], row["variant"]) for row in previous}
        if len(done) != len(previous):
            raise ValueError("duplicate result keys in output")
    else:
        done = set()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    null_cache = {}
    for case in cases:
        for variant in make_variants(case, "Unrelated: a violin is displayed in another building."):
            if (case["id"], variant["name"]) in done:
                continue
            row = evaluate_variant(model, tokenizer, variant, null_cache, torch)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(case["id"], variant["name"], row["predictions"], flush=True)


if __name__ == "__main__":
    main()
