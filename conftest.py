import os
from pathlib import Path

# Ensure subprocesses (e.g. integration tests spawning python -m ascii_combinator)
# can find the package regardless of their working directory.
PROJECT_ROOT = str(Path(__file__).parent)
existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{existing}" if existing else PROJECT_ROOT
