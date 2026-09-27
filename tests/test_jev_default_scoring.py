"""Real-model regression for the default RSS scoring mode.

Set FLINT_JEV_BIN, FLINT_JEV_MODEL, FLINT_JEV_TOKENIZER and
LD_LIBRARY_PATH to run; otherwise this optional integration test skips.
"""
import json
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DefaultScoringIntegration(unittest.TestCase):
    @unittest.skipUnless(all(os.getenv(key) for key in
                             ("FLINT_JEV_BIN", "FLINT_JEV_MODEL", "FLINT_JEV_TOKENIZER")),
                         "set local real-model paths")
    def test_default_uses_token_mean_and_explicit_total_remains_available(self):
        request = {"state": "The blue cup is in the kitchen. The red cup is in the bedroom.",
                   "questions": [{"id": "c", "kind": "choice", "prompt": "Where is the blue cup?",
                                  "choices": ["Kitchen", "Bedroom"]}]}
        base = [os.environ["FLINT_JEV_BIN"], "--llama", "--device", "cpu", "--script",
                "scripts/jev_lfm2_5.rss", "--", "--model", os.environ["FLINT_JEV_MODEL"],
                "--tokenizer", os.environ["FLINT_JEV_TOKENIZER"], "--request", json.dumps(request)]
        default = subprocess.run(base, cwd=ROOT, text=True, capture_output=True, timeout=120, check=True)
        total = subprocess.run(base + ["--scoring", "total"], cwd=ROOT, text=True,
                               capture_output=True, timeout=120, check=True)
        a, b = json.loads(default.stdout), json.loads(total.stdout)
        self.assertEqual(a["scoring"], "token-mean")
        self.assertEqual(b["scoring"], "total")
        self.assertTrue(all(e["ranking_score"] == e["log_likelihood"]
                            for e in b["results"][0]["distribution"]))
        self.assertTrue(any(e["ranking_score"] != e["log_likelihood"]
                            for e in a["results"][0]["distribution"]))


if __name__ == "__main__":
    unittest.main()
