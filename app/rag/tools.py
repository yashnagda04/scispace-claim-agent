"""The two tools exposed to the verification model."""

from app.rag.search import search

TOOL_DEFINITIONS = [
    {
        "type": "function", "name": "get_paper_info", "strict": True,
        "description": "Get a supplied paper's metadata, abstract, and indexed-text availability by paper_id.",
        "parameters": {
            "type": "object", "properties": {"paper_id": {"type": "string"}},
            "required": ["paper_id"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "search_evidence", "strict": True,
        "description": "Search up to twenty evidence passages. Use paper_ids=null for all indexed papers, a list to restrict papers, or [] for no matches. Refine queries if results are only topically related.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "paper_ids": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["query", "paper_ids"], "additionalProperties": False,
        },
    },
]


class ToolInputError(ValueError):
    """An invalid model argument; return it to the model and count the call."""


class RetrievalTools:
    def __init__(self, papers, collection, client, model, cohere_key=""):
        self.papers = {p["paper_id"]: p for p in papers}
        self.collection, self.client, self.model = collection, client, model
        self.cohere_key = cohere_key
        self.indexed = {}
        for metadata in collection.get(include=["metadatas"])["metadatas"]:
            pid = metadata["paper_id"]
            if pid not in self.papers:
                raise ValueError("Index contains paper IDs outside the prepared collection.")
            self.indexed.setdefault(pid, []).append(metadata)

    def get_paper_info(self, paper_id):
        if not isinstance(paper_id, str) or paper_id not in self.papers:
            raise ToolInputError("Unknown paper_id. Use an ID from the supplied collection.")
        paper = self.papers[paper_id]
        metadata = self.indexed.get(paper_id, [])
        return {
            "paper_id": paper_id, "title": paper["Paper Title"],
            "abstract": paper.get("Abstract", ""), "authors": paper.get("Author Names", ""),
            "year": paper.get("Publication Year", ""), "doi": paper.get("DOI", ""),
            "paper_link": paper.get("Paper Link", ""), "pdf_link": paper.get("PDF Link", ""),
            "indexed_text": {
                "chunk_count": len(metadata),
                "pdf": any(m["source_type"] == "pdf" for m in metadata),
                "abstract": any(m["source_type"] == "abstract" for m in metadata),
            },
        }

    def search_evidence(self, query, paper_ids=None):
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a nonempty string.")
        if paper_ids is not None:
            if not isinstance(paper_ids, list) or any(not isinstance(pid, str) for pid in paper_ids):
                raise ToolInputError("paper_ids must be a list of strings or null.")
            if set(paper_ids) - self.papers.keys():
                raise ToolInputError("Unknown paper_ids. The restricted search was not performed.")
            paper_ids = sorted(set(paper_ids) & self.indexed.keys())
            if not paper_ids:
                return {"passages": []}
        return {"passages": search(self.collection, self.client, self.model, query, paper_ids, self.cohere_key)}

    def execute(self, name, arguments):
        try:
            if not isinstance(arguments, dict):
                raise ToolInputError("Tool arguments must be a JSON object.")
            if name == "get_paper_info" and set(arguments) == {"paper_id"}:
                return self.get_paper_info(**arguments)
            if name == "search_evidence" and "query" in arguments and set(arguments) <= {"query", "paper_ids"}:
                return self.search_evidence(**arguments)
            raise ToolInputError("Unknown tool or invalid argument fields. Use the supplied tool schema.")
        except ToolInputError as exc:
            return {"error": str(exc)}
