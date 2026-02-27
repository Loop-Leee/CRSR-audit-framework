"""项目主入口。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunking.chunking_cli import main as run_chunking  # noqa: E402


if __name__ == "__main__":
    run_chunking()
