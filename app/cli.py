"""Run the local claim extraction and verification pipeline."""

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

from app.claim_extraction.extract import ExtractionIncomplete, extract_claims, paper_catalog
from app.claim_extraction.sections import split_report
from app.claim_extraction.storage import ClaimStore
from app.config import ROOT, Settings
from app.results.runs import create_run, update_run
from app.paper_preparation.prepare import prepare_papers, read_papers


def parser():
    result = argparse.ArgumentParser(description="Prepare papers, extract claims, build an index, verify, or export results.")
    result.add_argument("command", choices=["prepare", "extract", "prepare-rag", "verify", "export"])
    result.add_argument("--papers", type=Path, default=ROOT / "input_data/papers.csv")
    result.add_argument("--prepared", type=Path, default=ROOT / "data/papers.csv")
    result.add_argument("--report", type=Path, default=ROOT / "input_data/report.md")
    result.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    result.add_argument("--dry-run", action="store_true", help="Preview sections or PDF counts without network calls.")
    result.add_argument("--section-limit", type=int, help="Extract only the first N content sections for a sample run.")
    result.add_argument("--paper-ids", nargs="+", help="Prepare RAG for only these paper IDs, for a sample run.")
    result.add_argument("--claims", type=Path, help="Existing claims.csv to verify.")
    result.add_argument("--claim-limit", type=int, help="Verify the first N verifiable claims; overrides VERIFICATION_CLAIM_LIMIT.")
    result.add_argument("--run-dir", type=Path, help="Existing verification run to export without model calls.")
    return result


def run_extraction(args, settings, run, papers):
    if not args.dry_run:
        settings.require_openai()
    report = args.report.read_text(encoding="utf-8")
    if not report.strip():
        raise ValueError("Report is empty.")
    parsed_report = split_report(report)
    if not parsed_report.sections:
        raise ValueError("No content sections found in the report.")
    available_sections = len(parsed_report.sections)
    if args.section_limit is not None:
        parsed_report = replace(parsed_report, sections=parsed_report.sections[:args.section_limit])
    catalog = paper_catalog(papers)
    sections = [
        {"section_id": section.section_id, "section": section.heading}
        for section in parsed_report.sections
    ]
    update_run(
        run, sections=sections, available_section_count=available_sections,
        completed_sections=[], failed_sections=[], claim_count=0,
        report_sha256=hashlib.sha256(report.encode()).hexdigest(),
        papers_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
    )
    if args.dry_run:
        for section in parsed_report.sections:
            print(f"  {section.section_id}: {section.heading}")
        update_run(run, status="ready", dry_run=True, sections=sections)
        print(f"Selected {len(sections)} of {available_sections} sections. No model calls or claims created.")
        return 0
    store = ClaimStore(run)
    from openai import OpenAI

    with OpenAI(api_key=settings.api_key) as client:
        progress = extract_claims(
            client, settings, parsed_report, catalog, store,
            on_progress=lambda **values: update_run(run, **values),
        )
    unfinished = len(progress["failed_sections"])
    update_run(run, status="incomplete" if unfinished else "extraction_completed", **progress)
    print(f"Sections: {len(progress['completed_sections'])} completed, {unfinished} unfinished.")
    print(f"Saved {store.count} claims to {run / 'claims.csv'}")
    return 1 if unfinished else 0


def run_rag(args, settings, run, papers):
    if args.paper_ids:
        unknown = set(args.paper_ids) - {paper["paper_id"] for paper in papers}
        if unknown:
            raise ValueError(f"Unknown paper IDs: {', '.join(sorted(unknown))}.")
        papers = [paper for paper in papers if paper["paper_id"] in args.paper_ids]
    update_run(run, selected_paper_ids=[paper["paper_id"] for paper in papers])
    print(f"Selected papers for RAG: {len(papers)}")
    if args.dry_run:
        linked = sum(bool(paper.get("PDF Link", "").strip()) for paper in papers)
        print(f"PDF links: {linked}; skipped without a link: {len(papers) - linked}.")
        print(f"Embedding model: {settings.embedding_model}. No downloads or API calls made.")
        update_run(run, status="ready", dry_run=True)
        return 0
    settings.require_openai("embedding")
    from app.rag.ingest import prepare_rag

    result = prepare_rag(papers, settings, args.prepared.parent)
    failed = result["index"].get("failed", 0)
    update_run(run, status="incomplete" if failed else "rag_prepared", **result)
    print(f"Index: {result['index']}")
    return 1 if failed else 0


def run_verification(args, settings, run):
    from app.claim_verification.verify import verification_inputs, verify_claims
    from app.rag.search import open_collection
    from app.rag.tools import RetrievalTools
    from app.rag.rerank import CANDIDATE_LIMIT, RESULT_LIMIT, RERANK_MODEL

    _, papers = read_papers(args.prepared)
    claims, source = verification_inputs(args.claims, settings.verification_claim_limit,
                                          {p["paper_id"] for p in papers})
    collection = open_collection(args.prepared.parent / "vector_store", settings.embedding_model)
    tools = RetrievalTools(papers, collection, None, settings.embedding_model, settings.cohere_key)
    coverage = {"collection": collection.name, "indexed_paper_ids": sorted(tools.indexed),
                "indexed_paper_count": len(tools.indexed), "indexed_chunk_count": collection.count(),
                "metadata_paper_count": len(papers)}
    update_run(run, **source, index_coverage=coverage, prepared_papers=str(args.prepared.resolve()),
               retrieval={"candidate_limit": CANDIDATE_LIMIT if settings.cohere_key else RESULT_LIMIT,
                          "result_limit": RESULT_LIMIT,
                          "rerank_model": RERANK_MODEL if settings.cohere_key else None})
    print(f"Selected {len(claims)} claims; index covers {len(tools.indexed)} of {len(papers)} papers.")
    if args.dry_run:
        update_run(run, status="ready", dry_run=True)
        return 0
    settings.require_openai("verification")
    settings.require_openai("embedding")
    from openai import OpenAI

    with OpenAI(api_key=settings.api_key) as client:
        tools.client = client
        counts = verify_claims(client, settings, claims, tools, run)
    failed = counts["unfinished_count"]
    update_run(run, status="incomplete" if failed else "verification_completed", **counts)
    print(f"Verified: {counts['verified_count']}; unverified: {counts['unverified_count']}; unfinished: {failed}.")
    print(f"CSV: {run / 'results.csv'}")
    return 1 if failed else 0


def main(argv=None):
    args = parser().parse_args(argv)
    run = None
    try:
        if args.section_limit is not None and args.section_limit <= 0:
            raise ValueError("--section-limit must be a positive integer.")
        if args.paper_ids and args.command != "prepare-rag":
            raise ValueError("--paper-ids is only supported for prepare-rag.")
        if args.claim_limit is not None and args.claim_limit <= 0:
            raise ValueError("--claim-limit must be a positive integer.")
        if args.command == "export":
            if args.run_dir is None:
                raise ValueError("export requires --run-dir <verification run directory>.")
            from app.results.export import export_results

            print(f"Exported: {export_results(args.run_dir)}")
            return 0
        if args.command == "verify" and args.claims is None:
            raise ValueError("verify requires --claims <claims.csv>.")
        settings = Settings.from_env()
        if args.command == "verify" and args.claim_limit is not None:
            settings = replace(settings, verification_claim_limit=args.claim_limit)
        run = create_run(args.runs_dir, settings, args.command)
        print(f"Run: {run}")
        if args.command == "verify":
            return run_verification(args, settings, run)
        papers = prepare_papers(args.papers, args.prepared)
        update_run(run, paper_count=len(papers), prepared_papers=str(args.prepared.resolve()))
        print(f"Prepared papers: {len(papers)} ({args.prepared})")
        if args.command == "prepare":
            update_run(run, status="prepared")
            return 0
        if args.command == "prepare-rag":
            return run_rag(args, settings, run, papers)
        return run_extraction(args, settings, run, papers)
    except KeyboardInterrupt:
        if run:
            update_run(run, status="interrupted", error="Interrupted by user; accepted claims are preserved.")
        print("Interrupted; accepted claims are preserved.", file=sys.stderr)
        return 130
    except Exception as error:
        # Do not persist raw SDK exceptions, which can contain request data or keys.
        if isinstance(error, (ValueError, ExtractionIncomplete, FileNotFoundError)):
            message = str(error)
        else:
            message = f"{type(error).__name__}: processing failed; check dependencies, API access, and input files."
        status = "incomplete" if isinstance(error, ExtractionIncomplete) else "failed"
        if run:
            update_run(run, status=status, error=message)
        print(f"Error: {message}", file=sys.stderr)
        return 1
