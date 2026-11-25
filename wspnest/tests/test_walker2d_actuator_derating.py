"""Exercise the numeric patch contract and budget without loading MuJoCo."""
import json
from pathlib import Path
import sys
import unittest

TASK = Path(__file__).resolve().parents[1] / "tasks" / "walker2d-actuator-derating"
sys.path.insert(0, str(TASK))
try:
    from dev_server import BUDGET, Budget
    from model import parse_head
finally:
    sys.path.remove(str(TASK))


class WalkerContractTests(unittest.TestCase):
    def test_patch_bounds(self):
        for bad in (True, "0", float("inf"), float("nan"), 101):
            bad_head = [[bad] + [0] * 64] + [[0] * 65] * 5
            with self.assertRaises(ValueError):
                parse_head(json.dumps({"head": bad_head}).encode())
        self.assertEqual(parse_head(json.dumps({"head": [[0] * 65] * 6}).encode()).shape, (6, 65))

    def test_development_limit(self):
        seen = []

        def evaluate(head, block):
            seen.append(block)
            return [100, 200, 300]

        budget = Budget(evaluate)
        self.assertEqual(budget.query(b"{}")["remaining"], BUDGET)
        patch = json.dumps({"head": [[0] * 65] * 6}).encode()
        for i in range(BUDGET):
            self.assertEqual(budget.query(patch), {"mean_return": 200., "remaining": BUDGET - i - 1})
        self.assertIn("error", budget.query(patch))
        self.assertEqual(seen, [i // 13 for i in range(104)])
