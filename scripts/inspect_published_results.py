"""Read frozen v0.3 aggregate results without loading models or opening task answers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def inspect(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    base = report['absolute_metrics']['base']
    tuned = report['absolute_metrics']['dgp-regret']
    regret = base['dgp_final']['normalized_regret']
    improvement = (regret - tuned['dgp_final']['normalized_regret']) / regret
    if not math.isclose(improvement, report['metrics']['regret_relative_improvement']):
        raise ValueError('Published regret summary disagrees with absolute metrics')
    if report['ability_gates_passed'] != all(report['checks'].values()):
        raise ValueError('Published gate summary disagrees with its checks')
    return {
        'scope': 'Recomputed from published aggregates; no new model evaluation or bootstrap run.',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'regret_relative_improvement': improvement,
        'ability_gates_passed': report['ability_gates_passed'],
        'failed_gates': [key for key, passed in report['checks'].items() if not passed],
        'p_bench_strict_accuracy': [base['p_bench']['strict_accuracy'],
                                    tuned['p_bench']['strict_accuracy']],
        'statqa_exact_accuracy': [base['statqa']['exact_accuracy'],
                                   tuned['statqa']['exact_accuracy']],
        'clarification_accuracy': [base['clarification_accuracy'], tuned['clarification_accuracy']],
    }


if __name__ == '__main__':
    source = Path(__file__).resolve().parents[1] / 'reports/stats/evaluation.json'
    print(json.dumps(inspect(source), indent=2))
