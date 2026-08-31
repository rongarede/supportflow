from __future__ import annotations

import json
from pathlib import Path


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
    assert all(len(record["sha256"]) == 64 for record in manifest["records"].values())
    assert manifest["evaluation_report"]["path"].endswith(".json")
    assert len(manifest["evaluation_report"]["sha256"]) == 64
