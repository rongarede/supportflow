from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIRECTORY = PACKAGE_ROOT / "data" / "policies"
DEFAULT_RUNTIME_DIRECTORY = Path(".supportflow")


def runtime_database_path(runtime_directory: Path) -> Path:
    return runtime_directory / "supportflow.db"


def checkpoint_database_path(runtime_directory: Path) -> Path:
    return runtime_directory / "checkpoints.sqlite"
