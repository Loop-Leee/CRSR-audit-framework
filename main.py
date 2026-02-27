"""项目主入口。"""

from __future__ import annotations

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "classification":
        del sys.argv[1]
        from src.classification.classification_cli import main as run_classification

        run_classification()
    else:
        from src.chunking.chunking_cli import main as run_chunking

        run_chunking()
