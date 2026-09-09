"""Verify claims sequentially with two tools and a bounded retrieval budget."""

import csv
import json
import re
from pathlib import Path

from pydantic import ValidationError

from app.claim_verification.storage import Decision, EvidenceError, save_result, validate_evidence
from app.results.export import export_results
from app.results.runs import update_run
from app.rag.tools import TOOL_DEFINITIONS

JSON_FORMAT = {
    "type": "json_schema", "name": "verification_decision", "strict": True,
    "schema": Decision.model_json_schema(),
}


def read_claim_rows(path):
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "verifiable" not in (reader.fieldnames or []):
            raise ValueError("claims.csv is missing verifiable; rerun extraction with the current schema.")
        rows = list(reader)
    if any(row["verifiable"] not in {"true", "false"} for row in rows):
        raise ValueError("verifiable must be true or false in claims.csv.")
    return rows


def load_verifiable_claims(path, limit=20):
    """Take the first N verifiable claims; skipped rows do not use the limit."""
    return select_verifiable_claims(read_claim_rows(path), limit)


def select_verifiable_claims(rows, limit):
    if limit <= 0:
        raise ValueError("The verification claim limit must be positive.")
    return [{**row, "verifiable": True} for row in rows
            if row["verifiable"] == "true"][:limit]


def verification_inputs(path, limit, paper_ids):
    rows = read_claim_rows(path)
    selected = select_verifiable_claims(rows, limit)
    source_runs = {row.get("run_id", "") for row in rows}
    if len(source_runs) != 1 or "" in source_runs:
        raise ValueError("claims.csv must contain claims from one extraction run.")
    seen = set()
    for claim in selected:
        cid = claim.get("claim_id", "")
        if not re.fullmatch(r"claim_[0-9]+", cid) or cid in seen:
            raise ValueError("Claim IDs must be unique application-assigned claim_<number> IDs.")
        seen.add(cid)
        if not claim.get("claim_text", "").strip() or not claim.get("source_text", "").strip():
            raise ValueError("Each selected claim must contain claim_text and source_text.")
        try:
            ids = json.loads(claim["paper_ids"])
        except (KeyError, ValueError):
            raise ValueError("Claim paper_ids must contain a JSON list.") from None
        if not isinstance(ids, list) or any(not isinstance(pid, str) or pid not in paper_ids for pid in ids):
            raise ValueError("Claim paper_ids must come from the prepared collection.")
        claim["paper_ids"] = ids
    verifiable_count = sum(row["verifiable"] == "true" for row in rows)
    return selected, {
        "source_run_id": next(iter(source_runs)), "source_claims": str(Path(path).resolve()),
        "source_claim_count": len(rows), "skipped_non_verifiable": len(rows) - verifiable_count,
        "unselected_verifiable": verifiable_count - len(selected),
        "selected_claim_ids": [claim["claim_id"] for claim in selected],
    }


def retrieve_evidence(call, tools, sources):
    """Execute one permitted call and remember text available as evidence."""
    try:
        arguments = json.loads(call.arguments)
    except ValueError:
        return {"error": "Tool arguments must be valid JSON."}
    result = tools.execute(call.name, arguments)
    if call.name == "search_evidence":
        sources.extend(result.get("passages", []))
    elif call.name == "get_paper_info" and result.get("abstract", "").strip():
        sources.append({"paper_id": result["paper_id"], "text": result["abstract"], "location": "abstract"})
    return result


def verify_claim(client, settings, claim, tools, instructions, progress):
    messages = [{"role": "user", "content": json.dumps({
        "claim_text": claim["claim_text"], "source_text": claim["source_text"],
        "section": claim.get("section", ""), "paper_ids": claim["paper_ids"],
        "max_tool_calls": settings.max_tool_calls,
        "indexed_paper_ids": sorted(tools.indexed),
    }, ensure_ascii=False)}]
    sources = []
    # Each non-final response must use a tool; at most one final response follows the budget.
    for _ in range(settings.max_tool_calls + 1):
        final_only = progress["tool_calls"] >= settings.max_tool_calls
        response = client.responses.create(
            model=settings.verification_model, instructions=instructions,
            input=messages, tools=TOOL_DEFINITIONS, parallel_tool_calls=False,
            tool_choice="none" if final_only else ("required" if not progress["tool_calls"] else "auto"),
            text={"format": JSON_FORMAT}, include=["reasoning.encrypted_content"],
            store=False, truncation="disabled",
        )
        progress["model_requests"] += 1
        if response.usage:
            progress["input_tokens"] += response.usage.input_tokens
            progress["output_tokens"] += response.usage.output_tokens
        if response.status != "completed":
            raise ValueError("Verification response was incomplete.")
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            decision = Decision.model_validate_json(response.output_text)
            try:
                validate_evidence(decision, sources)
            except EvidenceError as exc:
                progress["evidence_error"] = str(exc)
                return Decision(verified=False, evidence=[],
                                reason="No valid supporting evidence was established.")
            return decision
        if final_only:
            raise ValueError("Model requested a tool after tools were disabled.")
        # Preserve reasoning items as well as function calls for the next request.
        messages.extend(response.output)
        for call in calls:
            if progress["tool_calls"] >= settings.max_tool_calls:
                result = {"error": "Retrieval budget exhausted; this call was not executed."}
            else:
                progress["tool_calls"] += 1
                result = retrieve_evidence(call, tools, sources)
            messages.append({"type": "function_call_output", "call_id": call.call_id,
                             "output": json.dumps(result, ensure_ascii=False)})
        messages.append({"role": "developer", "content": (
            f"Retrieval calls used: {progress['tool_calls']}/{settings.max_tool_calls}. "
            + ("No tool calls remain. Return your final JSON decision from the retrieved evidence. "
               "Use verified=false if support was not established."
               if progress["tool_calls"] >= settings.max_tool_calls else "Stop when you can make a supported decision.")
        )})
    raise ValueError("Verification ended without a final decision.")


def verify_claims(client, settings, claims, tools, run):
    instructions = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8")
    progress = []
    counts = {"verified_count": 0, "unverified_count": 0, "unfinished_count": 0}
    try:
        for claim in claims:
            entry = {"claim_id": claim["claim_id"], "status": "running", "tool_calls": 0,
                     "model_requests": 0, "input_tokens": 0, "output_tokens": 0}
            progress.append(entry)
            update_run(run, claim_progress=progress, **counts)
            print(f"Verifying {claim['claim_id']}...", flush=True)
            try:
                decision = verify_claim(client, settings, claim, tools, instructions, entry)
            except Exception as exc:
                # Keep API request contents and credentials out of logs.
                reason = "Invalid verification JSON." if isinstance(exc, ValidationError) else (
                    str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: verification failed."
                )
                entry.update(status="unfinished", reason=reason)
                counts["unfinished_count"] += 1
            else:
                save_result(run, claim, decision)
                entry.update(status="completed", verified=decision.verified)
                counts["verified_count" if decision.verified else "unverified_count"] += 1
            update_run(run, claim_progress=progress, **counts)
            export_results(run)
            print(f"{claim['claim_id']}: {entry['status']}, verified={entry.get('verified')}, tool calls={entry['tool_calls']}", flush=True)
    finally:
        update_run(run, claim_progress=progress, **counts)
        export_results(run)
    return counts
