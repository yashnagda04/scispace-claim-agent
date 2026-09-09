"""Download PDFs, extract evidence, and index it with OpenAI embeddings."""

import hashlib
import json
from importlib.metadata import version
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
import pymupdf4llm
from openai import OpenAI

from app.rag.download import download_pdfs
from app.rag.search import collection_name

EXTRACTOR = {
    "name": "pymupdf4llm",
    "version": version("pymupdf4llm"),
    "layout_version": version("pymupdf-layout"),
    "pymupdf_version": version("pymupdf"),
    "use_ocr": False,
}


def pdf_pages(path, cache_dir):
    """Keep page numbers and reuse extracted text when the PDF is unchanged."""
    path, cache_dir = Path(path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    cache_path = cache_dir / f"{path.stem}.json"
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["pdf_hash"] == digest and saved.get("extractor") == EXTRACTOR:
            return saved["pages"]
    # Layout analysis keeps columns in reading order. OCR stays off for this assignment.
    extracted = pymupdf4llm.to_markdown(str(path), page_chunks=True, use_ocr=False)
    pages = [
        {"page": page["metadata"]["page_number"], "text": page["text"]}
        for page in extracted
    ]
    cache_path.write_text(json.dumps({
        "pdf_hash": digest, "extractor": EXTRACTOR, "pages": pages,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return pages


def paper_chunks(paper, pages):
    sources = [{"source_type": "abstract", "page": 0, "text": paper.get("Abstract", "")}]
    sources.extend({"source_type": "pdf", **page} for page in pages)
    chunks = []
    for source in sources:
        text = source["text"].strip()
        # Small overlapping passages for retrieval, not limits on model output.
        for number, start in enumerate(range(0, len(text), 1800)):
            passage = text[start:start + 2000].strip()
            if not passage:
                continue
            chunks.append({
                "id": f"{paper['paper_id']}:{source['source_type']}:{source['page']}:{number}",
                "text": passage,
                "metadata": {
                    "paper_id": paper["paper_id"], "title": paper["Paper Title"],
                    "source_type": source["source_type"], "page": source["page"], "chunk": number,
                },
            })
    return chunks


def get_collection(vector_dir, model):
    # Separate collections prevent mixing vectors from different models.
    name = collection_name(model)
    return chromadb.PersistentClient(path=str(vector_dir)).get_or_create_collection(
        name=name, embedding_function=None, metadata={"embedding_model": model},
    )


def embed_chunks(client, model, chunks):
    vectors = []
    tokens = 0
    # Batching is required by the embedding endpoint's per-request input limit.
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        response = client.embeddings.create(
            model=model, input=[chunk["text"] for chunk in batch], encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(batch))):
            raise ValueError("Embedding response did not contain every requested passage.")
        vectors.extend(item.embedding for item in ordered)
        tokens += response.usage.total_tokens
    return vectors, tokens


def index_papers(papers, downloads, data_dir, collection, client, model, workers=4):
    data_dir = Path(data_dir)
    download_by_id = {item["paper_id"]: item for item in downloads}
    results = []
    pending = []
    for paper in papers:
        paper_id = paper["paper_id"]
        result = {"paper_id": paper_id}
        download = download_by_id[paper_id]
        pages = []
        if download["status"] in {"downloaded", "cached"}:
            try:
                pages = pdf_pages(download["path"], data_dir / "paper_text")
                result["pdf_text"] = "available" if any(p["text"].strip() for p in pages) else "no_extractable_text"
            except Exception as exc:
                result["pdf_text"] = f"unreadable: {type(exc).__name__}"
        else:
            result["pdf_text"] = download["status"]
        chunks = paper_chunks(paper, pages)
        result["chunk_count"] = len(chunks)
        existing = collection.get(where={"paper_id": paper_id}, include=["documents", "metadatas"])
        old = dict(zip(existing["ids"], zip(existing["documents"], existing["metadatas"])))
        desired = {chunk["id"]: (chunk["text"], chunk["metadata"]) for chunk in chunks}
        if not chunks:
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            result["status"] = "skipped_no_text"
        elif old == desired:
            result["status"] = "cached"
        else:
            result["status"] = "pending"
            pending.append((chunks, existing["ids"], result))
        results.append(result)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(embed_chunks, client, model, chunks): (chunks, old_ids, result)
            for chunks, old_ids, result in pending
        }
        for future in as_completed(futures):
            chunks, old_ids, result = futures[future]
            try:
                vectors, tokens = future.result()
                result["embedding_tokens"] = tokens
                # Keep Chroma writes on this thread while embedding calls run in parallel.
                collection.upsert(
                    ids=[chunk["id"] for chunk in chunks],
                    documents=[chunk["text"] for chunk in chunks],
                    metadatas=[chunk["metadata"] for chunk in chunks], embeddings=vectors,
                )
                stale = set(old_ids) - {chunk["id"] for chunk in chunks}
                if stale:
                    collection.delete(ids=list(stale))
                result["status"] = "indexed"
            except Exception as exc:
                # Changed papers must not leave outdated evidence in the index.
                if old_ids:
                    collection.delete(ids=old_ids)
                result.update(status="failed", reason=type(exc).__name__)
            print(f"Index {result['paper_id']}: {result['status']}", flush=True)

    (data_dir / "rag_status.json").write_text(json.dumps({
        "embedding_model": model, "collection": collection.name, "papers": results,
    }, indent=2) + "\n", encoding="utf-8")
    return results


def prepare_rag(papers, settings, data_dir):
    data_dir = Path(data_dir)
    print("Downloading linked PDFs with 8 workers...", flush=True)
    downloads = download_pdfs(papers, data_dir / "pdfs")
    download_counts = dict(Counter(item["status"] for item in downloads))
    print(f"PDFs: {download_counts}", flush=True)
    collection = get_collection(data_dir / "vector_store", settings.embedding_model)
    print(f"Embedding with {settings.embedding_model} using 4 workers...", flush=True)
    with OpenAI(api_key=settings.api_key) as client:
        indexed = index_papers(papers, downloads, data_dir, collection, client, settings.embedding_model)
    return {
        "downloads": download_counts,
        "index": dict(Counter(item["status"] for item in indexed)),
        "collection": collection.name, "embedding_model": settings.embedding_model,
        "embedding_tokens": sum(item.get("embedding_tokens", 0) for item in indexed),
    }
