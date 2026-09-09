"""Offline checks for Cohere ordering, failures, and credential handling."""

import unittest
from unittest.mock import patch

import httpx

from app.config import Settings
from app.rag.rerank import rerank


class RerankTests(unittest.TestCase):
    def test_short_and_empty_results_keep_original_text_and_metadata(self):
        passages = [{"text": "First", "paper_id": "A7K2", "location": "pdf page 2"},
                    {"text": "Second", "paper_id": "B2CD", "location": "abstract"}]
        with patch("app.rag.rerank.httpx.post", return_value=httpx.Response(
                200, json={"results": [{"index": 1}, {"index": 0}]})) as post:
            self.assertEqual(rerank("query", passages, "test-key"), passages[::-1])
            self.assertEqual(post.call_args.kwargs["json"]["top_n"], 2)
            self.assertEqual(post.call_args.kwargs["json"]["documents"], ["First", "Second"])
            self.assertEqual(rerank("query", [], "test-key"), [])
            self.assertEqual(post.call_count, 1)

    def test_invalid_results_and_http_failures_are_not_silently_accepted(self):
        passages = [{"text": "First"}, {"text": "Second"}]
        for indices in [[0, 0], [0, 2], [-1, 0], [True, 0], [0], ["0", 1]]:
            with self.subTest(indices=indices), patch("app.rag.rerank.httpx.post", return_value=
                    httpx.Response(200, json={"results": [{"index": i} for i in indices]})):
                with self.assertRaisesRegex(ValueError, "invalid rerank results"):
                    rerank("query", passages, "secret-key")
        for payload in [{}, {"results": None}, {"results": [{}]}, "invalid JSON"]:
            response = (httpx.Response(200, text=payload) if isinstance(payload, str)
                        else httpx.Response(200, json=payload))
            with patch("app.rag.rerank.httpx.post", return_value=response):
                with self.assertRaisesRegex(ValueError, "invalid rerank results"):
                    rerank("query", passages, "secret-key")
        for status in [401, 429, 503]:
            with patch("app.rag.rerank.httpx.post", return_value=httpx.Response(status, text="secret-key")):
                with self.assertRaisesRegex(ValueError, f"Cohere rerank failed \\(HTTP {status}\\)"):
                    rerank("query", passages, "secret-key")
        with patch("app.rag.rerank.httpx.post", side_effect=httpx.ReadTimeout("secret-key")):
            with self.assertRaisesRegex(ValueError, r"^Cohere rerank request failed \(ReadTimeout\)\.$"):
                rerank("query", passages, "secret-key")

    def test_cohere_key_is_loaded_but_never_in_public_settings(self):
        with patch("app.config.load_dotenv"), patch.dict("os.environ", {"COHERE_KEY": " secret-key "}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.cohere_key, "secret-key")
        self.assertNotIn("cohere_key", settings.public_dict())
        self.assertNotIn("secret-key", repr(settings))


if __name__ == "__main__":
    unittest.main()
