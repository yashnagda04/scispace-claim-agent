"""Recovery must fetch the same paper and retain the original failure."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.rag.download import download_pdfs
from app.rag.sources import PMC_CLOUD, pmc_pdf_url
from test_rag import paper, sample_pdf


class PdfRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paper = {**paper(url="https://scispace.com/pdf/study.pdf"), "DOI": "10.1234/study"}

    def cloud_response(self, request, **changes):
        if request.url.path == "/":
            # Version 1 need not exist; do not invent its URL.
            return httpx.Response(200, text='''<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
                <IsTruncated>false</IsTruncated><CommonPrefixes><Prefix>PMC123.2/</Prefix></CommonPrefixes>
                </ListBucketResult>''')
        if request.url.path.endswith(".json"):
            return httpx.Response(200, json={
                "pmcid": "PMC123", "title": "A study.", "doi": "10.1234/STUDY",
                "pdf_url": "s3://pmc-oa-opendata/PMC123.2/PMC123.2.pdf?md5=example",
                "is_manuscript": False, **changes,
            })
        return httpx.Response(200, content=sample_pdf())

    def test_blocked_pdf_recovers_by_doi_with_original_failure_preserved(self):
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.host == "scispace.com":
                return httpx.Response(403)
            if request.url.host == "pmc.ncbi.nlm.nih.gov":
                self.assertEqual(request.url.params["ids"], "10.1234/study")
                return httpx.Response(200, json={"status": "ok", "records": [
                    {"doi": "10.1234/study", "pmcid": "PMC123"},
                ]})
            return self.cloud_response(request)

        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch("app.rag.download.httpx.Client", return_value=client):
            result = download_pdfs([self.paper], self.root)[0]
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(result["source"], "pmc")
        self.assertEqual(result["url"], self.paper["PDF Link"])
        self.assertEqual(result["attempts"][0]["http_status"], 403)
        self.assertEqual(result["resolved_url"], f"{PMC_CLOUD}PMC123.2/PMC123.2.pdf")
        self.assertEqual(sum(r.url.host == "scispace.com" for r in requests), 1)
        self.assertTrue((self.root / "A7K2.pdf").is_file())

    def test_html_challenge_recovers_by_explicit_pmc_link_without_doi_lookup(self):
        def handle(request):
            if request.url.host == "www.ncbi.nlm.nih.gov":
                return httpx.Response(200, text="<html>Preparing to download ...</html>")
            self.assertEqual(request.url.host, "pmc-oa-opendata.s3.amazonaws.com")
            return self.cloud_response(request)

        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch("app.rag.download.httpx.Client", return_value=client):
            result = download_pdfs([paper(url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/pdf")], self.root)[0]
        self.assertEqual(result["status"], "downloaded")
        self.assertIn("browser challenge", result["attempts"][0]["reason"])

    def test_wrong_identity_and_missing_pdf_are_rejected(self):
        for changes in [{"title": "An unrelated study"}, {"doi": "10.1234/other"},
                        {"pmcid": "PMC456"}, {"pdf_url": None},
                        {"pdf_url": "s3://other-bucket/PMC123.2/fake.pdf"}]:
            with self.subTest(changes=changes):
                with httpx.Client(transport=httpx.MockTransport(
                        lambda request: self.cloud_response(request, **changes))) as client:
                    with self.assertRaises(ValueError):
                        pmc_pdf_url(self.paper, "PMC123", client)
        self.assertFalse(list(self.root.glob("*.pdf")))

    def test_transient_timeout_retries_once_and_cleans_partial_file(self):
        calls = []

        def handle(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout("slow server", request=request)
            return httpx.Response(200, content=sample_pdf())

        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch("app.rag.download.httpx.Client", return_value=client):
            result = download_pdfs([self.paper], self.root)[0]
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(len(calls), 2)
        self.assertFalse(list(self.root.glob("*.part")))

    def test_failed_lookup_is_reported_without_accepting_a_pdf(self):
        def handle(request):
            if request.url.host == "scispace.com":
                return httpx.Response(403)
            return httpx.Response(503)

        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch("app.rag.download.httpx.Client", return_value=client):
            result = download_pdfs([self.paper], self.root)[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("lookup failed", result["fallback_reason"])
        self.assertFalse(list(self.root.glob("*.pdf")))
