"""One JSON extraction request per report section, with one correction attempt."""

import json
from pathlib import Path

from pydantic import ValidationError

from app.claim_extraction.sections import Report, Section
from app.claim_extraction.storage import ClaimStore, SectionClaims, validate_claims
from app.config import Settings

PROMPT_PATH = Path(__file__).with_name("prompt.md")
JSON_FORMAT = {
    "type": "json_schema", "name": "section_claims", "strict": True,
    "schema": SectionClaims.model_json_schema(),
}


class ExtractionIncomplete(RuntimeError):
    """A section cannot be completed; previously saved results remain available."""


def paper_catalog(papers):
    return [{"paper_id": paper["paper_id"], "title": paper["Paper Title"]} for paper in papers]


def section_input(report: Report, section: Section, catalog):
    return json.dumps({
        "report_title": report.title,
        "paper_catalog": catalog,
        "bibliography": report.bibliography,
        "section": section.text,
    }, ensure_ascii=False, separators=(",", ":"))


def extract_section(client, settings, instructions, request_input, section, paper_ids):
    messages = [{"role": "user", "content": request_input}]
    for attempt in range(2):
        response = client.responses.create(
            model=settings.extraction_model, instructions=instructions, input=messages,
            text={"format": JSON_FORMAT},
            truncation="disabled", store=False,
        )
        try:
            if response.status != "completed":
                raise ValueError("Response was incomplete. Return a complete JSON object.")
            if not response.output_text:
                raise ValueError("No JSON claims returned; the response may have been refused.")
            batch = SectionClaims.model_validate_json(response.output_text)
            validate_claims(batch, section, paper_ids)
            return batch
        except ValidationError:
            error = "Invalid JSON structure. Return exactly the claims schema with all required fields and correct types."
        except ValueError as exc:
            error = str(exc)
        if attempt == 0:
            messages.extend([
                {"role": "assistant", "content": response.output_text or "No JSON returned."},
                {"role": "user", "content": f"{error} Correct the response for the same section. Return all its claims again."},
            ])
    raise ExtractionIncomplete(error)


def extract_claims(client, settings: Settings, report: Report, catalog, store: ClaimStore, on_progress=None):
    instructions = PROMPT_PATH.read_text(encoding="utf-8")
    paper_ids = {paper["paper_id"] for paper in catalog}
    completed = []
    failures = []
    for section in report.sections:
        try:
            batch = extract_section(
                client, settings, instructions, section_input(report, section, catalog), section, paper_ids,
            )
        except ExtractionIncomplete as exc:
            failures.append({"section_id": section.section_id, "section": section.heading, "reason": str(exc)})
        else:
            store.save_section(section, batch)
            completed.append(section.section_id)
        if on_progress:
            on_progress(claim_count=store.count, completed_sections=list(completed), failed_sections=list(failures))
    return {"claim_count": store.count, "completed_sections": completed, "failed_sections": failures}
