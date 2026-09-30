"""HandTap2PDScore: finger-tapping video -> hand keypoints -> kinematics -> PD / MDS-UPDRS 3.4 score."""
from pathlib import Path

__version__ = "0.1.0"

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
# Weights live next to the package so the same layout works on the cluster and in a HF Space.
MODELS = Path(__import__("os").environ.get("HANDTAP_MODELS", ROOT / "models"))
