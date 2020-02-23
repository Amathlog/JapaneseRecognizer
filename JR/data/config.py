import os
from pathlib import Path

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR = CURRENT_DIR / ".." / "data_temp"