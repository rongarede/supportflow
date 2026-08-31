import hashlib
import json

from supportflow.domain.models import ResolutionProposal


def proposal_hash(proposal: ResolutionProposal) -> str:
    payload = proposal.model_dump(mode="json", exclude={"proposal_hash"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
