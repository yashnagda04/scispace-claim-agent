"""Rebuild the final CSV from saved verification JSON; no model calls."""

import csv
import json
from pathlib import Path

from app.claim_verification.storage import VerificationResult

FIELDS = ["run_id", "claim_id", "claim_text", "verified", "evidence", "reason"]


def export_results(run):
    run = Path(run)
    if not (run / "verification").is_dir():
        raise ValueError("The run directory has no verification folder.")
    records = []
    for path in sorted((run / "verification").glob("*.json")):
        record = VerificationResult.model_validate_json(path.read_text(encoding="utf-8"))
        if record.run_id != run.name or path.stem != record.claim_id:
            raise ValueError("Verification JSON does not match its run or claim ID.")
        records.append(record.model_dump())
    # Preserve selection order when metadata is available.
    metadata_path = run / "run.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    order = {cid: i for i, cid in enumerate(metadata.get("selected_claim_ids", []))}
    records.sort(key=lambda row: (order.get(row["claim_id"], len(order)), row["claim_id"]))
    destination = run / "results.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "verified": json.dumps(row["verified"]),
                             "evidence": json.dumps(row["evidence"], ensure_ascii=False)})
    temporary.replace(destination)
    return destination
