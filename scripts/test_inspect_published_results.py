"""Check published-summary arithmetic and preserve negative outcomes."""
import json
import tempfile
import unittest
from pathlib import Path

from inspect_published_results import inspect

SOURCE = Path(__file__).resolve().parents[1] / 'reports/stats/evaluation.json'


class PublishedResultsTests(unittest.TestCase):
    def test_negative_outcomes_remain_visible(self):
        result = inspect(SOURCE)
        self.assertFalse(result['ability_gates_passed'])
        self.assertEqual(result['failed_gates'], ['p_bench', 'statqa'])
        self.assertAlmostEqual(result['regret_relative_improvement'], 0.3404208893470001)
        self.assertEqual(result['clarification_accuracy'][1], 0)

    def test_inconsistent_summary_is_rejected(self):
        for field in ['metrics', 'ability_gates_passed']:
            report = json.loads(SOURCE.read_text())
            if field == 'metrics':
                report[field]['regret_relative_improvement'] = 0.99
            else:
                report[field] = True
            with tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / 'report.json'
                file.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    inspect(file)


if __name__ == '__main__':
    unittest.main()
