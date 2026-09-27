import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from jev_accuracy_eval import aggregate, score_methods, paired_summary, make_variants, render_prefix, validate_cases, continuation_ids, aggregate_both_orders


class AccuracyEvalTests(unittest.TestCase):
    def test_system_instruction_can_change_without_reordering_choices(self):
        case = {"state": "Six minus two is four.", "prompt": "How many remain?",
                "choices": ["Four", "Six"], "system": "Compute the result before selecting. Output only the selected answer."}
        prefix = render_prefix(case, "raw")
        self.assertIn(case["system"] + "<|im_end|>", prefix)
        self.assertIn("- Four\n- Six\n", prefix)

    def test_dual_order_aligns_semantic_scores(self):
        pair = {
            "original": {"choices": ["Blue", "Red"], "scores": {"raw": {"Blue": -1, "Red": -2}}},
            "reversed": {"choices": ["Red", "Blue"], "scores": {"raw": {"Red": -3, "Blue": -1}}},
        }
        self.assertEqual(aggregate_both_orders(pair, "raw"), "Blue")
        with self.assertRaises(ValueError):
            aggregate_both_orders({"original": pair["original"]}, "raw")

    def test_continuation_requires_token_prefix_and_nonempty_suffix(self):
        class Tokenizer:
            def encode(self, value, add_special_tokens=False):
                return [ord(x) for x in value]
        tokenizer = Tokenizer()
        self.assertEqual(continuation_ids(tokenizer, "Hi", " yes"), ([72, 105], [32, 121, 101, 115]))
        with self.assertRaises(ValueError):
            continuation_ids(tokenizer, "Hi", "")

    def test_prompt_modes_preserve_semantic_choices(self):
        case = {"state": "The box is closed.", "prompt": "Is it open?", "choices": ["No", "Yes"]}
        raw = render_prefix(case, "raw")
        letters = render_prefix(case, "letter")
        self.assertIn("- No\n- Yes\n", raw)
        self.assertIn("A. No\nB. Yes\n", letters)
        self.assertTrue(letters.endswith("Answer:"))
        self.assertTrue(raw.startswith("<|startoftext|>"))
        self.assertEqual(render_prefix({**case, "state": "N/A"}, "raw").count("The box is closed."), 0)

    def test_case_validation_rejects_wrong_answer_and_duplicate_ids(self):
        case = {"id": "a", "state": "S", "prompt": "Q?", "paraphrase": "P?", "choices": ["Yes", "No"], "gold": "Yes", "tag": "lookup"}
        validate_cases([case])
        with self.assertRaises(ValueError):
            validate_cases([{**case, "gold": "Maybe"}])
        with self.assertRaises(ValueError):
            validate_cases([case, case])

    def test_scoring_keeps_single_token_and_length_priors_distinct(self):
        answers = [
            {"label": "Long answer", "token_logprobs": [-2.0, -0.1], "null_logprob": -3.0, "null_tokens": 2, "letter_logprob": -1.0},
            {"label": "Short", "token_logprobs": [-1.2], "null_logprob": -0.1, "null_tokens": 1, "letter_logprob": -2.0},
        ]
        scores = score_methods(answers)
        self.assertEqual(aggregate(scores["raw"]), "Short")
        self.assertEqual(aggregate(scores["mean"]), "Long answer")
        self.assertEqual(aggregate(scores["pmi"]), "Long answer")
        self.assertEqual(aggregate(scores["letter"]), "Long answer")

    def test_permutation_is_mapped_to_semantic_label(self):
        case = {"id": "x", "state": "Pip is a cat", "prompt": "Species?", "paraphrase": "Which animal?", "choices": ["Cat", "Dog"], "gold": "Cat"}
        variants = make_variants(case, "Unrelated: the window is open.")
        self.assertEqual([v["name"] for v in variants], ["original", "reversed", "paraphrased", "distractor"])
        self.assertEqual(variants[1]["choices"], ["Dog", "Cat"])
        self.assertEqual(variants[1]["gold"], "Cat")
        self.assertEqual(variants[2]["prompt"], "Which animal?")
        self.assertIn("window is open", variants[3]["state"])

    def test_paired_summary_counts_each_answer_without_tuning(self):
        rows = [
            {"id": "a", "gold": "Yes", "tag": "lookup", "predictions": {"raw": "Yes", "letter": "No"}},
            {"id": "b", "gold": "No", "tag": "unknown", "predictions": {"raw": "Yes", "letter": "No"}},
        ]
        summary = paired_summary(rows)
        self.assertEqual(summary["raw"]["correct"], 1)
        self.assertEqual(summary["letter"]["correct"], 1)
        self.assertEqual(summary["raw"]["by_tag"]["unknown"], {"correct": 0, "total": 1})


if __name__ == "__main__":
    unittest.main()
