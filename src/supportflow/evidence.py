from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HASH_ALGORITHM = "SHA-256 of UTF-8 canonical JSON (sorted keys, compact separators)"
MANIFEST_VERSION = "supportflow-evidence-manifest/v1"


def canonical_json(payload: Any) -> str:
    """Serialize a JSON-safe, sanitized payload in one reproducible form."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def hashed_record(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a verifiable digest to a payload that has already been sanitized."""
    return {"source": source, "payload": payload, "sha256": canonical_sha256(payload)}


def build_manifest(
    *,
    source_revision: str,
    run: dict[str, Any],
    records: dict[str, dict[str, Any]],
    evaluation_path: str,
    evaluation_payload: Any,
) -> dict[str, Any]:
    """Build a portable manifest without copying secrets or raw customer content."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "sanitized": True,
        "hash_algorithm": HASH_ALGORITHM,
        "source_revision": source_revision,
        "run": run,
        "records": records,
        "evaluation_report": {
            "path": evaluation_path,
            "sha256": canonical_sha256(evaluation_payload),
        },
    }


def export_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the deterministic, reviewable JSON artifact after callers sanitize inputs."""
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
