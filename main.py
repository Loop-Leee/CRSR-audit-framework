"""项目主入口。"""

from __future__ import annotations

import sys

from src.chunking.chunking_cli import main as run_chunking
from src.classification.classification_cli import main as run_classification


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "classification":
        del sys.argv[1]
        run_classification()
    else:
        run_chunking()
