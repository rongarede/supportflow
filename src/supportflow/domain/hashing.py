import hashlib
import json

from supportflow.domain.models import ResolutionProposal, Ticket


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_input_revision(ticket: Ticket) -> str:
    """Derive a stable revision from source content, not local ingestion time."""
    return _canonical_sha256(
        ticket.model_dump(mode="json", exclude={"created_at"})
    )


def proposal_hash(proposal: ResolutionProposal) -> str:
    payload = proposal.model_dump(mode="json", exclude={"proposal_hash"})
    return _canonical_sha256(payload)
