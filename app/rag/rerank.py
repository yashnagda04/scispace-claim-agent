"""Use Cohere to select the most relevant original passages."""

import httpx

RERANK_MODEL = "rerank-v4.0-pro"
CANDIDATE_LIMIT = 100
RESULT_LIMIT = 20


def rerank(query, passages, api_key):
    if not passages:
        return []
    top_n = min(RESULT_LIMIT, len(passages))
    try:
        response = httpx.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": RERANK_MODEL, "query": query,
                  "documents": [p["text"] for p in passages], "top_n": top_n},
            timeout=60,
        )
    except httpx.HTTPError as exc:
        raise ValueError(f"Cohere rerank request failed ({type(exc).__name__}).") from None
    if not response.is_success:
        raise ValueError(f"Cohere rerank failed (HTTP {response.status_code}).")
    try:
        results = response.json()["results"]
        indices = [result["index"] for result in results]
        if (len(indices) != top_n
                or any(type(i) is not int or not 0 <= i < len(passages) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("Cohere returned invalid rerank results.") from None
    # Keep our exact text and metadata; Cohere only supplies the ordering.
    return [passages[i] for i in indices]
