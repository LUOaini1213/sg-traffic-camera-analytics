import os
import sys
from pathlib import Path

# TCAM_SRC lets scripts/mutation_check.py point the tests at a mutated copy of the package.
sys.path.insert(0, os.environ.get("TCAM_SRC", str(Path(__file__).resolve().parents[1] / "src")))
