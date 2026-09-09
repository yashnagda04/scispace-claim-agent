"""Download listed PDF links concurrently and reuse files already on disk."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from pypdf import PdfReader

from app.rag.sources import find_pmc_ids, pmc_pdf_url


def is_pdf(path):
    try:
        with Path(path).open("rb") as stream:
            if b"%PDF-" not in stream.read(1024):
                return False
            PdfReader(stream)
        return True
    except Exception:
        return False


def fetch_pdf(url, path, client):
    """Save only a readable PDF; retry one transient network/server failure."""
    temporary = path.with_suffix(".pdf.part")
    details = {"url": url}
    for attempt in range(2):
        try:
            with client.stream("GET", url) as response:
                details.update(resolved_url=str(response.url), http_status=response.status_code,
                               content_type=response.headers.get("content-type", ""))
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_bytes():
                        stream.write(block)
            if not is_pdf(temporary):
                with temporary.open("rb") as stream:
                    sample = stream.read(4096).decode("utf-8", errors="replace").lower()
                challenge = any(marker in sample for marker in (
                    "preparing to download", "checking your browser", "client challenge", "not a robot",
                ))
                reason = "Site returned a browser challenge instead of a PDF." if challenge else "The link did not return a readable PDF."
                return {**details, "status": "failed", "reason": reason}
            temporary.replace(path)
            return {**details, "status": "downloaded", "path": str(path)}
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {500, 502, 503, 504} and attempt == 0:
                continue
            reason = f"HTTP {status}"
            if status in {401, 403, 418}:
                reason += " (site denied automated access)"
            return {**details, "status": "failed", "reason": reason}
        except httpx.TransportError as exc:
            if attempt == 0:
                continue
            return {**details, "status": "failed", "reason": type(exc).__name__}
        except Exception as exc:
            return {**details, "status": "failed", "reason": type(exc).__name__}
        finally:
            temporary.unlink(missing_ok=True)


def download_one(paper, pdf_dir, client):
    paper_id = paper["paper_id"]
    url = paper.get("PDF Link", "").strip()
    result = {"paper_id": paper_id, "url": url}
    if not url:
        return {**result, "status": "skipped_no_link"}
    path = pdf_dir / f"{paper_id}.pdf"
    if path.exists() and is_pdf(path):
        return {**result, "status": "cached", "path": str(path)}
    attempt = fetch_pdf(url, path, client)
    return {**result, **attempt, "attempts": [attempt]}


def recover_pdf(paper, result, pmcid, client, pdf_dir):
    try:
        url = pmc_pdf_url(paper, pmcid, client)
        attempt = fetch_pdf(url, pdf_dir / f"{paper['paper_id']}.pdf", client)
        result["attempts"].append(attempt)
        if attempt["status"] == "downloaded":
            result.update(status="downloaded", path=attempt["path"], resolved_url=attempt["resolved_url"],
                          http_status=attempt["http_status"], content_type=attempt["content_type"], source="pmc", pmcid=pmcid)
            result.pop("reason", None)
        else:
            result["fallback_reason"] = attempt["reason"]
    except Exception as exc:
        result["fallback_reason"] = str(exc) if isinstance(exc, ValueError) else f"PMC fallback failed: {type(exc).__name__}."
    return result


def download_pdfs(papers, pdf_dir, workers=8):
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    # Slow PDF servers need longer than HTTPX's five-second default.
    with httpx.Client(follow_redirects=True, timeout=30,
                      headers={"User-Agent": "SciSpaceAssignment/1.0"}) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda paper: download_one(paper, pdf_dir, client), papers))
            failed = [paper for paper, result in zip(papers, results) if result["status"] == "failed"]
            pmc_ids, lookup_error = find_pmc_ids(failed, client)
            futures = []
            for paper, result in zip(papers, results):
                if result["status"] != "failed":
                    continue
                pmcid = pmc_ids.get(paper["paper_id"])
                if pmcid:
                    futures.append(pool.submit(recover_pdf, paper, result, pmcid, client, pdf_dir))
                else:
                    result["fallback_reason"] = lookup_error or "No PMC match for the supplied DOI or PDF link."
            for future in futures:
                future.result()
    (pdf_dir / "downloads.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
