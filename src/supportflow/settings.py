from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIRECTORY = PACKAGE_ROOT / "data" / "policies"
