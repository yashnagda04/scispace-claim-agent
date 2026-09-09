"""Validate decisions and save one result per claim."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paper_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    location: str = Field(min_length=1)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verified: bool
    evidence: list[Evidence]
    reason: str = Field(min_length=1)


class VerificationResult(Decision):
    run_id: str
    claim_id: str
    claim_text: str


class EvidenceError(ValueError):
    """The supplied evidence cannot establish verified support."""


def validate_evidence(decision, sources):
    if not decision.reason.strip():
        raise ValueError("The decision must include a short reason.")
    if decision.verified and not decision.evidence:
        raise EvidenceError("A verified claim must have retrieved evidence.")
    for evidence in decision.evidence:
        if not evidence.text.strip() or not any(
            evidence.paper_id == source["paper_id"]
            and evidence.location == source["location"]
            and evidence.text in source["text"]
            for source in sources
        ):
            raise EvidenceError("Evidence must quote a returned passage with its matching paper ID and location.")


def save_result(run, claim, decision):
    result = VerificationResult(
        run_id=Path(run).name, claim_id=claim["claim_id"], claim_text=claim["claim_text"],
        **decision.model_dump(),
    )
    path = Path(run) / "verification" / f"{claim['claim_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result
