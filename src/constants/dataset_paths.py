"""Dataset directory paths for the calibration and evaluation subsets."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CALIBRATION_DIR: Path = PROJECT_ROOT / "data" / "raw" / "NYC_Calibration_100"
EVALUATION_DIR: Path = PROJECT_ROOT / "data" / "raw" / "NYC_Evaluation_1000"

SAN_FRAN_CENTRE_CALIBRATION_DIR: Path = PROJECT_ROOT / "data" / "raw" / "SanFranCentre_Calibration_100"
SAN_FRAN_CENTRE_EVALUATION_DIR: Path = PROJECT_ROOT / "data" / "raw" / "SanFranCentre_Evaluation_1000"
