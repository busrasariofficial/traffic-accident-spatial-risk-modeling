import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = PROJECT_ROOT / "app" / "app.py"


subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(APP_FILE)
])