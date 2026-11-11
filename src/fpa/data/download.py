"""Download the M5 files with the Kaggle CLI (SPEC §2.2)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

REQUIRED_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")

MANUAL_INSTRUCTIONS = """
Could not download M5 automatically. To get the data manually:
  1. Sign in at https://www.kaggle.com and accept the competition rules:
       https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
  2. Either
       a) create an API token (Kaggle > Settings > API > Create New Token), save it as
          ~/.kaggle/kaggle.json (chmod 600), then re-run:  python run.py download
     or
       b) download the zip from
          https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
          and unzip it into: {raw_dir}
  Required files: {files}
"""


def has_raw_data(raw_dir: Path) -> bool:
    """True if all required M5 CSVs are present."""
    return all((raw_dir / f).exists() for f in REQUIRED_FILES)


def download(raw_dir: Path, competition: str) -> bool:
    """Download and unzip M5 into `raw_dir`. Returns True on success."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if has_raw_data(raw_dir):
        log.info("M5 files already present in %s", raw_dir)
        return True
    if shutil.which("kaggle") is None:
        print(MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, files=", ".join(REQUIRED_FILES)))
        return False
    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(raw_dir)]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        log.error("Kaggle download failed: %s", exc)
        print(MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, files=", ".join(REQUIRED_FILES)))
        return False
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
    ok = has_raw_data(raw_dir)
    if not ok:
        print(MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, files=", ".join(REQUIRED_FILES)))
    return ok
