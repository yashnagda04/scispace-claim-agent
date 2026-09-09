"""Validate section JSON, assign claim IDs, and save JSON plus CSV."""

import csv
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.claim_extraction.sections import Section


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claim_text: str
    source_text: str
    verifiable: bool
    reference_numbers: list[int]
    paper_ids: list[str]


class SectionClaims(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claims: list[Claim]


CSV_FIELDS = [
    "run_id", "claim_id", "section_id", "section",
    "claim_text", "source_text", "verifiable", "reference_numbers", "paper_ids",
]


def validate_claims(batch: SectionClaims, section: Section, paper_ids: set[str]):
    """Validate the entire response before writing any part of it."""
    for index, claim in enumerate(batch.claims, start=1):
        if not claim.claim_text.strip() or not claim.source_text.strip():
            raise ValueError(f"Claim {index}: claim_text and source_text must be non-empty.")
        if claim.source_text not in section.text:
            raise ValueError(f"Claim {index}: source_text must be copied exactly from the current section.")
        if any(number <= 0 for number in claim.reference_numbers):
            raise ValueError(f"Claim {index}: reference_numbers must contain positive integers.")
        if any(paper_id not in paper_ids for paper_id in claim.paper_ids):
            raise ValueError(f"Claim {index}: paper_ids must come from the supplied title-and-ID catalog.")


class ClaimStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.json_dir = self.run_dir / "extraction"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "claims.csv"
        self.count = 0
        self.saved_sections = set()
        # A store belongs to a new run. Never overwrite a previous run's claims.
        with self.csv_path.open("x", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()

    def save_section(self, section: Section, batch: SectionClaims):
        """Save an already validated section once, including empty results."""
        if section.section_id in self.saved_sections:
            return
        rows = []
        seen = set()
        for claim in batch.claims:
            data = claim.model_dump()
            data["paper_ids"] = sorted(set(data["paper_ids"]))
            data["reference_numbers"] = sorted(set(data["reference_numbers"]))
            signature = json.dumps(data, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append({
                "run_id": self.run_dir.name,
                "claim_id": f"claim_{self.count + len(rows) + 1:04d}",
                "section_id": section.section_id,
                "section": section.heading,
                **data,
            })

        result = {
            "run_id": self.run_dir.name, "section_id": section.section_id,
            "section": section.heading, "claims": rows,
        }
        path = self.json_dir / f"{section.section_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            for row in rows:
                writer.writerow({**row,
                    "verifiable": json.dumps(row["verifiable"]),
                    "reference_numbers": json.dumps(row["reference_numbers"]),
                    "paper_ids": json.dumps(row["paper_ids"]),
                })
        self.count += len(rows)
        self.saved_sections.add(section.section_id)
