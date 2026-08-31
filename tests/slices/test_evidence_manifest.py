from __future__ import annotations

import json
from pathlib import Path

from supportflow.evidence import build_manifest, canonical_sha256, hashed_record

def test_committed_evidence_manifest_has_versioned_sanitized_hashes() -> None:
    """Catches portfolio evidence that points only at a mutable ignored runtime database."""
    manifest = json.loads(Path("artifacts/evidence-manifest-v1.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "supportflow-evidence-manifest/v1"
    assert manifest["sanitized"] is True
    assert manifest["run"]["run_id"]
    assert set(manifest["records"]) == {
        "ticket",
        "triage",
        "evidence_bundle",
        "proposal",
        "risk_review",
        "policy_decision",
        "approval",
        "execution_results",
        "checkpoint",
        "trace",
    }
    assert manifest["hash_algorithm"] == "SHA-256 of UTF-8 canonical JSON (sorted keys, compact separators)"
    assert all(record["sha256"] == canonical_sha256(record["payload"]) for record in manifest["records"].values())
    assert manifest["evaluation_report"]["path"].endswith(".json")
    evaluation = json.loads(Path(manifest["evaluation_report"]["path"]).read_text(encoding="utf-8"))
    assert manifest["evaluation_report"]["sha256"] == canonical_sha256(evaluation)


def test_committed_manifest_is_rebuilt_by_the_deterministic_exporter() -> None:
    """Catches a hand-authored manifest that cannot be reproduced by the exporter."""
    manifest = json.loads(Path("artifacts/evidence-manifest-v1.json").read_text(encoding="utf-8"))
    evaluation = json.loads(Path(manifest["evaluation_report"]["path"]).read_text(encoding="utf-8"))

    rebuilt = build_manifest(
        source_revision=manifest["source_revision"],
        run=manifest["run"],
        records={
            name: hashed_record(record["source"], record["payload"])
            for name, record in manifest["records"].items()
        },
        evaluation_path=manifest["evaluation_report"]["path"],
        evaluation_payload=evaluation,
    )

    assert rebuilt == manifest
