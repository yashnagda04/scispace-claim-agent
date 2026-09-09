"""Open an existing Chroma index and retrieve a few evidence passages."""

import hashlib
from pathlib import Path

import chromadb

from app.rag.rerank import CANDIDATE_LIMIT, RESULT_LIMIT, rerank


def collection_name(model):
    return "papers_" + hashlib.sha256(model.encode()).hexdigest()[:12]


def open_collection(vector_dir, model):
    if not Path(vector_dir).is_dir():
        raise ValueError("No vector index found. Run prepare-rag first.")
    client = chromadb.PersistentClient(path=str(vector_dir))
    try:
        collection = client.get_collection(collection_name(model), embedding_function=None)
    except chromadb.errors.NotFoundError:
        raise ValueError("No index for this embedding model. Run prepare-rag first.") from None
    if collection.metadata.get("embedding_model") != model:
        raise ValueError("The index embedding model does not match the configured model.")
    if not collection.count():
        raise ValueError("The vector index is empty. Run prepare-rag first.")
    return collection


def search(collection, client, model, query, paper_ids=None, cohere_key=""):
    where = {"paper_id": {"$in": paper_ids}} if paper_ids is not None else None
    response = client.embeddings.create(model=model, input=[query], encoding_format="float")
    found = collection.query(
        query_embeddings=[response.data[0].embedding], where=where,
        n_results=min(CANDIDATE_LIMIT if cohere_key else RESULT_LIMIT, collection.count()),
        include=["documents", "metadatas"],
    )
    passages = []
    for chunk_id, text, metadata in zip(found["ids"][0], found["documents"][0], found["metadatas"][0]):
        location = f"pdf page {metadata['page']}" if metadata["source_type"] == "pdf" else "abstract"
        passages.append({"chunk_id": chunk_id, "text": text, **metadata, "location": location})
    return rerank(query, passages, cohere_key) if cohere_key else passages
