"""项目主入口。"""

from __future__ import annotations

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "classification":
        del sys.argv[1]
        from src.classification.classification_cli import main as run_classification

        run_classification()
    elif len(sys.argv) > 1 and sys.argv[1] == "baseline":
        del sys.argv[1]
        from src.baseline.baseline_cli import main as run_baseline

        run_baseline()
    elif len(sys.argv) > 1 and sys.argv[1] == "review":
        del sys.argv[1]
        from src.review.review_cli import main as run_review

        run_review()
    elif len(sys.argv) > 1 and sys.argv[1] == "result":
        del sys.argv[1]
        from src.result.result_cli import main as run_result

        run_result()
    elif len(sys.argv) > 1 and sys.argv[1] == "reflection":
        del sys.argv[1]
        from src.reflection.reflection_cli import main as run_reflection

        run_reflection()
    elif len(sys.argv) > 1 and sys.argv[1] == "experiment":
        del sys.argv[1]
        from src.experiment.run_experiment import main as run_experiment

        run_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "chunking-ablation":
        del sys.argv[1]
        from src.chunking.chunking_ablation_cli import main as run_chunking_ablation

        run_chunking_ablation()
    else:
        from src.chunking.chunking_cli import main as run_chunking

        run_chunking()
