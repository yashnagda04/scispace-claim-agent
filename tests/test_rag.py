"""Offline PDF and Chroma checks; HTTP and OpenAI requests are mocked."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import Settings
from main import main
from app.paper_preparation.prepare import prepare_papers
from app.rag.download import download_pdfs
from app.rag.ingest import embed_chunks, get_collection, index_papers, pdf_pages


def sample_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 200 Td (Measured result in the paper.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_blank_page(width=300, height=300)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def paper(paper_id="A7K2", abstract="A measured result.", url=""):
    return {"paper_id": paper_id, "Paper Title": "A study", "Abstract": abstract,
            "PDF Link": url, "Summary": "Generated summary is not evidence."}


class Embeddings:
    def __init__(self, barrier=None):
        self.embeddings = self
        self.requests = []
        self.barrier = barrier

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.barrier:
            self.barrier.wait(timeout=10)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=len(kwargs["input"])), data=[
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0, 0.5])
            for i, text in reversed(list(enumerate(kwargs["input"])))
        ])


class RagTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_cli_sample_only_passes_requested_papers_to_rag(self):
        source = self.root / "source.csv"
        source.write_text("Paper Title,Abstract,PDF Link\nOne,Text,https://example.test/one\nTwo,Text,https://example.test/two\n")
        prepared = self.root / "papers.csv"
        papers = prepare_papers(source, prepared)
        args = ["prepare-rag", "--papers", str(source), "--prepared", str(prepared),
                "--runs-dir", str(self.root / "runs"), "--paper-ids", papers[1]["paper_id"]]
        with patch("app.cli.Settings.from_env", return_value=Settings(api_key="test")), \
                patch("app.rag.ingest.prepare_rag", return_value={"index": {"cached": 1}}) as prepare, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 0)
            self.assertEqual(prepare.call_args.args[0], [papers[1]])
            prepare.reset_mock()
            self.assertEqual(main([*args[:-1], "UNKNOWN"]), 1)
            prepare.assert_not_called()

        records = [json.loads(p.read_text()) for p in (self.root / "runs").glob("*/run.json")]
        success = next(r for r in records if r["status"] == "rag_prepared")
        self.assertEqual(success["selected_paper_ids"], [papers[1]["paper_id"]])

    def test_downloads_parallel_validate_and_reuse_pdfs(self):
        barrier = Barrier(2)
        requests = []
        pdf = sample_pdf()

        def handle(request):
            requests.append(request.url.path)
            if request.url.path in {"/one", "/two"}:
                barrier.wait(timeout=10)  # Fails if downloads run sequentially.
                return httpx.Response(200, content=pdf)
            if request.url.path == "/html":
                return httpx.Response(200, text="<html>Sign in</html>")
            return httpx.Response(404)

        papers = [paper("A1BC", url="https://example.test/one"),
                  paper("A2BC", url="https://example.test/two"),
                  paper("A3BC", url="https://example.test/html"),
                  paper("A4BC", url="https://example.test/missing"), paper("A5BC")]
        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch("app.rag.download.httpx.Client", return_value=client):
            results = download_pdfs(papers, self.root, workers=2)
        self.assertEqual([r["status"] for r in results],
                         ["downloaded", "downloaded", "failed", "failed", "skipped_no_link"])
        self.assertEqual(results[3]["reason"], "HTTP 404")
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(list(self.root.glob("*.pdf"))), 2)
        self.assertEqual(list(self.root.glob("*.part")), [])
        self.assertEqual(json.loads((self.root / "downloads.json").read_text()), results)

        def unexpected_request(request):
            self.fail("Cached PDFs must not be downloaded again.")

        client = httpx.Client(transport=httpx.MockTransport(unexpected_request))
        with patch("app.rag.download.httpx.Client", return_value=client):
            reused = download_pdfs(papers[:2], self.root)
        self.assertEqual([r["status"] for r in reused], ["cached", "cached"])

    def test_pdf_text_and_page_numbers_are_cached(self):
        path = self.root / "A7K2.pdf"
        path.write_bytes(sample_pdf())
        pages = pdf_pages(path, self.root / "text")
        self.assertIn("Measured result", pages[0]["text"])
        self.assertEqual(pages[0]["page"], 1)
        self.assertEqual(pages[1], {"page": 2, "text": ""})
        with patch("app.rag.ingest.pymupdf4llm.to_markdown", side_effect=AssertionError("PDF was reparsed")):
            self.assertEqual(pdf_pages(path, self.root / "text"), pages)

        cache_path = self.root / "text/A7K2.json"
        saved = json.loads(cache_path.read_text())
        saved.pop("extractor")  # Old pypdf cache must not survive the extractor change.
        saved["pages"][0]["text"] = "Outdated extraction"
        cache_path.write_text(json.dumps(saved))
        self.assertEqual(pdf_pages(path, self.root / "text"), pages)

    def test_parallel_embedding_index_query_and_cache(self):
        path = self.root / "A7K2.pdf"
        path.write_bytes(sample_pdf())
        papers = [paper(), paper("B2CD"), paper("C3DE", abstract="")]
        downloads = [{"paper_id": "A7K2", "status": "downloaded", "path": str(path)},
                     {"paper_id": "B2CD", "status": "failed"},
                     {"paper_id": "C3DE", "status": "skipped_no_link"}]
        model = Settings().embedding_model
        collection = get_collection(self.root / "vectors", model)
        client = Embeddings(Barrier(2))
        results = index_papers(papers, downloads, self.root, collection, client, model, workers=2)
        self.assertEqual([r["status"] for r in results], ["indexed", "indexed", "skipped_no_text"])
        self.assertEqual(collection.count(), 3)
        self.assertEqual(len(client.requests), 2)
        self.assertTrue(all(request["model"] == "text-embedding-3-small" for request in client.requests))
        found = collection.query(query_embeddings=[[20.0, 1.0, 0.5]],
                                 where={"paper_id": "A7K2"}, n_results=2)
        self.assertEqual({m["source_type"] for m in found["metadatas"][0]}, {"abstract", "pdf"})
        self.assertTrue(all(m["paper_id"] == "A7K2" for m in found["metadatas"][0]))
        self.assertFalse(any("Generated summary" in text for text in found["documents"][0]))
        again = get_collection(self.root / "vectors", model)
        reused = index_papers(papers, downloads, self.root, again, client, model)
        self.assertEqual([r["status"] for r in reused], ["cached", "cached", "skipped_no_text"])
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(get_collection(self.root / "vectors", "different-model").count(), 0)

    def test_changed_content_replaces_chunks_and_failure_removes_old_evidence(self):
        model = Settings().embedding_model
        collection = get_collection(self.root / "vectors", model)
        downloads = [{"paper_id": "A7K2", "status": "skipped_no_link"}]
        client = Embeddings()
        index_papers([paper(abstract="long passage " * 400)], downloads, self.root, collection, client, model)
        self.assertGreater(collection.count(), 1)
        index_papers([paper(abstract="Short new abstract.")], downloads, self.root, collection, client, model)
        self.assertEqual(collection.count(), 1)
        self.assertEqual(collection.get()["documents"], ["Short new abstract."])
        with patch.object(client, "create", side_effect=RuntimeError("API unavailable")):
            failed = index_papers([paper(abstract="Changed again.")], downloads, self.root, collection, client, model)
        self.assertEqual(failed[0]["status"], "failed")
        self.assertEqual(collection.count(), 0)

    def test_embedding_batches_and_response_order(self):
        client = Embeddings()
        chunks = [{"text": "x" * n} for n in range(1, 36)]
        vectors, tokens = embed_chunks(client, "text-embedding-3-small", chunks)
        self.assertEqual(tokens, 35)
        self.assertEqual([len(request["input"]) for request in client.requests], [32, 3])
        self.assertEqual([vector[0] for vector in vectors], list(range(1, 36)))


if __name__ == "__main__":
    unittest.main()
