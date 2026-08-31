from __future__ import annotations

import json
from pathlib import Path

import pytest

from supportflow.evidence import (
    canonical_sha256,
    generate_manifest_from_deterministic_run,
    source_revision,
)

def test_committed_evidence_manifest_has_versioned_sanitized_hashes() -> None:
    """Catches portfolio evidence that points only at a mutable ignored runtime database."""
    manifest = json.loads(Path("artifacts/evidence-manifest-v1.json").read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == "supportflow-evidence-manifest/v1"
    assert manifest["source_revision"] == source_revision(Path.cwd())
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
    evidence = manifest["records"]["evidence_bundle"]["payload"]
    assert evidence["query"]
    assert evidence["sufficient"] is True
    assert evidence["unresolved_questions"] == []
    assert 1 <= len(evidence["active_evidence"]) <= 5


def test_committed_manifest_is_rebuilt_from_a_fresh_deterministic_run(
    tmp_path, monkeypatch
) -> None:
    """Catches an exporter that copies the committed manifest instead of runtime evidence."""
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    manifest = json.loads(Path("artifacts/evidence-manifest-v1.json").read_text(encoding="utf-8"))
    rebuilt = generate_manifest_from_deterministic_run(
        repo_root=Path.cwd(),
        runtime_directory=tmp_path / "evidence-export",
        evaluation_path=Path(manifest["evaluation_report"]["path"]),
        supplied_source_revision=manifest["source_revision"],
    )

    assert rebuilt == manifest


def test_export_rejects_a_source_revision_not_bound_to_the_executable_tree(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    with pytest.raises(ValueError, match="does not match"):
        generate_manifest_from_deterministic_run(
            repo_root=Path.cwd(),
            runtime_directory=tmp_path / "evidence-export",
            evaluation_path=next(Path("artifacts").glob("eval-*.json")),
            supplied_source_revision="supportflow-source-sha256:not-current",
        )
