from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationSummary:
    dataset_sha256: str
    route_accuracy: float
    required_evidence_hit_rate: float
    citation_validity: float
    action_accuracy: float
    terminal_state_accuracy: float
    unapproved_execution_count: int
    duplicate_side_effect_count: int
    recovery_failure_count: int
    policy_sha256: str
    case_count: int


def write_reports(
    summary: EvaluationSummary, results: list[dict[str, Any]], output_dir: Path, adapters: dict[str, str]
) -> tuple[Path, Path]:
    """Write stable report names and content for a dataset/policy input pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "adapters": adapters,
        "audit": {
            "dataset_sha256": summary.dataset_sha256,
            "policy_sha256": summary.policy_sha256,
            "evidence_pointers": "Per-case observed.evidence_ids are policy chunk evidence pointers.",
            "bad_case_categories": ["missing_information", "policy_conflict", "duplicate_submission"],
            "scope": "Deterministic fixture wiring and safety evidence, not real-LLM quality.",
        },
        "cases": results,
    }
    json_path = output_dir / f"eval-{summary.dataset_sha256}.json"
    markdown_path = output_dir / f"eval-{summary.dataset_sha256}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
                "# SupportFlow frozen evaluation",
                "",
                f"- Dataset SHA-256: `{summary.dataset_sha256}`",
                f"- Policy SHA-256: `{summary.policy_sha256}`",
                f"- Adapters: `{adapters['model']}` + `{adapters['embedding']}`",
                f"- Cases: {summary.case_count}",
                f"- Route accuracy: {summary.route_accuracy:.3f}",
                f"- Required-evidence hit rate: {summary.required_evidence_hit_rate:.3f}",
                f"- Citation validity: {summary.citation_validity:.3f}",
                f"- Action accuracy: {summary.action_accuracy:.3f}",
                f"- Terminal-state accuracy: {summary.terminal_state_accuracy:.3f}",
                f"- Unapproved executions: {summary.unapproved_execution_count}",
                f"- Duplicate side effects: {summary.duplicate_side_effect_count}",
                f"- Recovery failures: {summary.recovery_failure_count}",
                "",
                "All actions are simulated. These are deterministic fixture wiring/safety evidence, not real-LLM quality, a baseline, or an uplift claim.",
                "",
                "## Per-case audit",
                "",
                "| Case | Route | Evidence pointers | Proposed actions | Terminal state | Score flags |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
    for result in results:
        observed = result["observed"]
        score_flags = ", ".join(
            key for key in ("route_correct", "citations_valid", "action_correct", "terminal_state_correct") if result[key]
        )
        lines.append(
            f"| {result['case_id']} | {observed['route'] or '-'} | {', '.join(observed['evidence_ids']) or '-'} | {', '.join(observed['proposed_actions']) or '-'} | {observed['terminal_state']} | {score_flags or '-'} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
