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
    summary: EvaluationSummary, results: list[dict[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    """Write stable report names and content for a dataset/policy input pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": asdict(summary), "cases": results}
    json_path = output_dir / f"eval-{summary.dataset_sha256}.json"
    markdown_path = output_dir / f"eval-{summary.dataset_sha256}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(
        "\n".join(
            [
                "# SupportFlow frozen evaluation",
                "",
                f"- Dataset SHA-256: `{summary.dataset_sha256}`",
                f"- Policy SHA-256: `{summary.policy_sha256}`",
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
                "All actions are simulated. These are measured fixture results, not a baseline or uplift claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path
