"""Offline retrieval, tool budget, provenance, SDK loop, and export tests."""

import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import httpx
from openai import OpenAI

from app.claim_verification.storage import Decision, save_result, validate_evidence
from app.claim_verification.verify import verify_claim, verify_claims
from app.config import Settings
from main import main
from app.results.export import export_results
from app.results.runs import create_run, update_run
from app.rag.search import collection_name, open_collection
from app.rag.tools import RetrievalTools

TEXT = 'In adults, the device improved adherence by 20% after 12 weeks.\nNo longer follow-up was available.'
EVIDENCE = {"paper_id": "A7K2", "text": TEXT.splitlines()[0], "location": "abstract"}
CLAIM = {"claim_id": "claim_0001", "claim_text": "Adult adherence improved by 20% at 12 weeks.",
         "source_text": "Adult adherence improved by 20% at 12 weeks [1].", "paper_ids": ["A7K2"]}


def decision(verified=True, **changes):
    return {"verified": verified, "evidence": [EVIDENCE] if verified else [],
            "reason": "Supported by the reported result." if verified else "Support was not established.", **changes}


def call(name="get_paper_info", arguments=None, cid="call_1"):
    return SimpleNamespace(type="function_call", name=name, call_id=cid,
                           arguments=json.dumps(arguments or {"paper_id": "A7K2"}))


def response(*calls, final=None, status="completed", raw=None):
    return SimpleNamespace(output=list(calls), status=status, usage=None,
                           output_text=raw if raw is not None else json.dumps(final) if final else "")


def progress():
    return {"tool_calls": 0, "model_requests": 0, "input_tokens": 0, "output_tokens": 0}


class Client:
    def __init__(self, responses=()):
        self.responses = self
        self.queue = iter(responses)
        self.requests = []
        self.embedding_requests = []
        self.embeddings = SimpleNamespace(create=self.embed)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        result = next(self.queue)
        if isinstance(result, Exception):
            raise result
        return result

    def embed(self, **kwargs):
        self.embedding_requests.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])])


class VerificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.settings = Settings()
        self.papers = [
            {"paper_id": "A7K2", "Paper Title": "Adherence study", "Abstract": TEXT,
             "Adherence Findings": "Generated information must not be returned."},
            {"paper_id": "B2CD", "Paper Title": "Another study", "Abstract": "Different findings."},
            {"paper_id": "C3DE", "Paper Title": "Unindexed study", "Abstract": "Local abstract."},
        ]
        self.collection = chromadb.PersistentClient(path=str(self.root / "vectors")).create_collection(
            name=collection_name(self.settings.embedding_model), embedding_function=None,
            metadata={"embedding_model": self.settings.embedding_model},
        )
        self.collection.add(
            ids=["A7K2:pdf:2:0", "B2CD:pdf:3:0"],
            documents=[TEXT, "Different findings."], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[{"paper_id": pid, "title": title, "source_type": "pdf", "page": page, "chunk": 0}
                       for pid, title, page in [("A7K2", "Adherence study", 2), ("B2CD", "Another study", 3)]],
        )

    def tools(self, client):
        return RetrievalTools(self.papers, self.collection, client, self.settings.embedding_model)

    def test_tools_filter_and_never_broaden_invalid_or_empty_search(self):
        client = Client()
        tools = self.tools(client)
        info = tools.get_paper_info("A7K2")
        self.assertEqual(info["abstract"], TEXT)
        self.assertNotIn("Adherence Findings", info)
        self.assertTrue(info["indexed_text"]["pdf"])
        self.assertFalse(tools.get_paper_info("C3DE")["indexed_text"]["pdf"])
        for ids, expected in [(None, {"A7K2", "B2CD"}), (["A7K2"], {"A7K2"}),
                              (["A7K2", "B2CD"], {"A7K2", "B2CD"})]:
            found = tools.search_evidence("adherence", ids)["passages"]
            self.assertEqual({p["paper_id"] for p in found}, expected)
            self.assertTrue(all(p["location"] == f"pdf page {p['page']}" for p in found))
        before = len(client.embedding_requests)
        for ids in [[], ["C3DE"]]:
            self.assertEqual(tools.search_evidence("adherence", ids), {"passages": []})
        for args in [{"query": "adherence", "paper_ids": ["A7K2", "UNKNOWN"]},
                     {"query": "adherence", "paper_ids": "A7K2"}, {"query": " "}]:
            self.assertIn("error", tools.execute("search_evidence", args))
        self.assertEqual(len(client.embedding_requests), before)
        self.assertEqual(open_collection(self.root / "vectors", self.settings.embedding_model).count(), 2)
        with self.assertRaisesRegex(ValueError, "No index for"):
            open_collection(self.root / "vectors", "another-model")

    def test_search_includes_twelfth_ranked_evidence_and_caps_at_twenty(self):
        ids = [f"A7K2:pdf:{i + 4}:0" for i in range(25)]
        self.collection.add(
            ids=ids, documents=[f"Passage {i}: {TEXT}" for i in range(25)],
            embeddings=[[1.0, (i + 1) / 100, 0.0] for i in range(25)],
            metadatas=[{"paper_id": "A7K2", "title": "Adherence study", "source_type": "pdf",
                        "page": i + 4, "chunk": 0} for i in range(25)],
        )
        tools = self.tools(Client())
        for paper_ids in [None, ["A7K2"], ["A7K2", "B2CD"]]:
            with self.subTest(paper_ids=paper_ids):
                passages = tools.search_evidence("adherence", paper_ids)["passages"]
                self.assertEqual(len(passages), 20)
                self.assertEqual(len({p["chunk_id"] for p in passages}), 20)
                self.assertEqual(passages[11], {
                    "chunk_id": ids[10], "text": f"Passage 10: {TEXT}", "paper_id": "A7K2",
                    "title": "Adherence study", "source_type": "pdf", "page": 14,
                    "chunk": 0, "location": "pdf page 14",
                })
        restricted = tools.search_evidence("adherence", ["B2CD"])["passages"]
        self.assertEqual([p["chunk_id"] for p in restricted], ["B2CD:pdf:3:0"])

    def test_verified_and_qualifier_mismatch_decisions(self):
        for final in [decision(), decision(False, reason="The source reports 12 weeks, not 12 months.")]:
            client = Client([response(call()), response(final=final)])
            result = verify_claim(client, self.settings, CLAIM, self.tools(client), "test prompt", progress())
            self.assertEqual(result.model_dump(), final)
            self.assertEqual(len(client.requests), 2)
            self.assertFalse(client.requests[0]["parallel_tool_calls"])
            self.assertNotIn("max_output_tokens", client.requests[0])

    def test_cohere_uses_100_filtered_candidates_and_returns_20_original_passages(self):
        ids = [f"A7K2:pdf:{i + 4}:0" for i in range(105)]
        self.collection.add(
            ids=ids, documents=[f"Passage {i}" for i in range(105)],
            embeddings=[[1.0, (i + 1) / 100, 0.0] for i in range(105)],
            metadatas=[{"paper_id": "A7K2", "source_type": "pdf", "page": i + 4,
                        "chunk": 0, "title": "Adherence study"} for i in range(105)],
        )
        client = Client()
        tools = RetrievalTools(self.papers, self.collection, client,
                               self.settings.embedding_model, "test-cohere-key")
        with patch("app.rag.rerank.httpx.post", return_value=httpx.Response(
                200, json={"results": [{"index": i} for i in [80, *range(19)]]})) as post:
            found = tools.search_evidence("adherence", ["A7K2"])["passages"]
            self.assertEqual(len(post.call_args.kwargs["json"]["documents"]), 100)
            self.assertEqual(post.call_args.kwargs["json"]["top_n"], 20)
            self.assertEqual(len(found), 20)
            self.assertEqual({p["paper_id"] for p in found}, {"A7K2"})
            self.assertEqual(found[0], {"chunk_id": ids[79], "text": "Passage 79",
                                      "paper_id": "A7K2", "source_type": "pdf", "page": 83,
                                      "chunk": 0, "title": "Adherence study", "location": "pdf page 83"})
        with patch("app.rag.rerank.httpx.post", return_value=httpx.Response(
                200, json={"results": [{"index": 0}]})) as post:
            found = tools.search_evidence("adherence", ["B2CD"])["passages"]
            self.assertEqual([p["paper_id"] for p in found], ["B2CD"])
            self.assertEqual(post.call_args.kwargs["json"]["documents"], ["Different findings."])
            before = len(client.embedding_requests)
            for paper_ids in [[], ["C3DE"]]:
                self.assertEqual(tools.search_evidence("adherence", paper_ids), {"passages": []})
            self.assertIn("error", tools.execute("search_evidence", {"query": "adherence", "paper_ids": ["A7K2", "UNKNOWN"]}))
            self.assertEqual(len(client.embedding_requests), before)
            self.assertEqual(post.call_count, 1)

    def test_cohere_failure_is_unfinished_and_counts_the_tool_call(self):
        run = create_run(self.root / "runs", self.settings, "verify")
        client = Client([response(call(name="search_evidence", arguments={"query": "adherence"}))])
        tools = RetrievalTools(self.papers, self.collection, client,
                               self.settings.embedding_model, "test-cohere-key")
        with patch("app.rag.rerank.httpx.post", return_value=httpx.Response(429)), redirect_stdout(io.StringIO()):
            counts = verify_claims(client, self.settings, [CLAIM], tools, run)
        self.assertEqual(counts, {"verified_count": 0, "unverified_count": 0, "unfinished_count": 1})
        record = json.loads((run / "run.json").read_text())
        self.assertEqual(record["claim_progress"][0]["tool_calls"], 1)
        self.assertEqual(record["claim_progress"][0]["reason"], "Cohere rerank failed (HTTP 429).")
        self.assertEqual(list((run / "verification").glob("*.json")), [])

    def test_budget_allows_final_decision_and_never_executes_21st_call(self):
        for supported in [True, False]:
            final = decision(supported, reason="Supported." if supported else "Budget exhausted; support not established.")
            calls = [response(call(arguments={"paper_id": "UNKNOWN"}, cid=f"call_{i}")) for i in range(19)]
            calls.append(response(call(arguments={"paper_id": "A7K2" if supported else "UNKNOWN"}, cid="last")))
            client = Client([*calls, response(final=final)])
            state = progress()
            result = verify_claim(client, self.settings, CLAIM, self.tools(client), "prompt", state)
            self.assertEqual(state["tool_calls"], 20)
            self.assertEqual(len(client.requests), 21)
            self.assertEqual(client.requests[-1]["tool_choice"], "none")
            self.assertEqual(result.verified, supported)

        client = Client([response(*[call(cid=f"c{i}") for i in range(25)]), response(final=decision())])
        state = progress()
        tools = self.tools(client)
        with patch.object(tools, "execute", wraps=tools.execute) as execute:
            verify_claim(client, self.settings, CLAIM, tools, "prompt", state)
            self.assertEqual(execute.call_count, 20)
        self.assertEqual(state["tool_calls"], 20)
        self.assertEqual(client.requests[-1]["tool_choice"], "none")

    def test_invalid_calls_count_and_missing_evidence_returns_false(self):
        malformed = call()
        malformed.arguments = "{invalid"
        client = Client([response(malformed), response(call(arguments={"paper_id": "UNKNOWN"})),
                         response(call(name="search_evidence", arguments={"query": "missing", "paper_ids": []})),
                         response(final=decision(False))])
        state = progress()
        result = verify_claim(client, self.settings, CLAIM, self.tools(client), "prompt", state)
        self.assertFalse(result.verified)
        self.assertEqual(state["tool_calls"], 3)
        self.assertIn('"error"', client.requests[1]["input"][-2]["output"])

    def test_fabricated_quotes_and_wrong_locations_are_rejected(self):
        sources = [{**EVIDENCE, "text": TEXT}]
        for evidence in [[], [{**EVIDENCE, "text": "Invented finding."}],
                         [{**EVIDENCE, "paper_id": "B2CD"}], [{**EVIDENCE, "location": "pdf page 2"}]]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate_evidence(Decision(**decision(evidence=evidence)), sources)

    def test_failures_remain_unfinished_and_other_claims_export(self):
        cases = [response(raw="bad json"), response(status="incomplete"), response(raw=""),
                 RuntimeError("private API details"), response(final=decision(False, reason=" "))]
        for i, bad in enumerate(cases):
            run = create_run(self.root / str(i), self.settings, "verify")
            claims = [CLAIM, {**CLAIM, "claim_id": "claim_0002"}]
            client = Client([bad, response(call()), response(final=decision(False))])
            with redirect_stdout(io.StringIO()):
                counts = verify_claims(client, self.settings, claims, self.tools(client), run)
            self.assertEqual(counts, {"verified_count": 0, "unverified_count": 1, "unfinished_count": 1})
            self.assertFalse((run / "verification/claim_0001.json").exists())
            with (run / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["claim_id"] for r in rows], ["claim_0002"])
            self.assertEqual(len(client.requests[1]["input"]), 1)
            self.assertNotIn("private API details", (run / "run.json").read_text())

    def test_invalid_evidence_saves_false_and_continues_without_retry(self):
        invalid = [[], [{**EVIDENCE, "text": "Invented finding."}],
                   [{**EVIDENCE, "paper_id": "B2CD"}], [{**EVIDENCE, "location": "pdf page 2"}]]
        for i, evidence in enumerate(invalid):
            for claimed_verified in [True, False]:
                if not claimed_verified and not evidence:
                    continue  # False with no evidence is already a valid decision.
                with self.subTest(evidence=evidence, verified=claimed_verified):
                    run = create_run(self.root / str(i), self.settings, "verify")
                    claims = [CLAIM, {**CLAIM, "claim_id": "claim_0002"}]
                    client = Client([response(call()), response(final=decision(claimed_verified, evidence=evidence)),
                                     response(call()), response(final=decision())])
                    with redirect_stdout(io.StringIO()):
                        counts = verify_claims(client, self.settings, claims, self.tools(client), run)
                    self.assertEqual(counts, {"verified_count": 1, "unverified_count": 1, "unfinished_count": 0})
                    record = json.loads((run / "verification/claim_0001.json").read_text())
                    self.assertFalse(record["verified"])
                    self.assertEqual(record["evidence"], [])
                    self.assertEqual(record["reason"], "No valid supporting evidence was established.")
                    metadata = json.loads((run / "run.json").read_text())
                    self.assertIn("evidence_error", metadata["claim_progress"][0])
                    self.assertEqual(len(client.requests), 4)
                    with (run / "results.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual([row["verified"] for row in rows], ["false", "true"])

    def test_sdk_round_trip_preserves_reasoning_and_call_outputs(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            output = ([{"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"},
                       {"id": "fc_1", "type": "function_call", "call_id": "c1", "name": "get_paper_info",
                        "arguments": '{"paper_id":"A7K2"}', "status": "completed"}]
                      if len(requests) == 1 else
                      [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(decision()), "annotations": []}]}])
            return httpx.Response(200, json={"id": f"resp_{len(requests)}", "object": "response", "created_at": 0,
                                            "model": "test", "status": "completed", "output": output})

        with OpenAI(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            result = verify_claim(client, self.settings, CLAIM, self.tools(client), "prompt", progress())
        self.assertTrue(result.verified)
        self.assertEqual(requests[1]["input"][1]["encrypted_content"], "opaque")
        self.assertEqual(requests[1]["input"][3]["call_id"], "c1")
        self.assertEqual(requests[0]["model"], "gpt-5.6-terra")
        self.assertEqual(requests[0]["include"], ["reasoning.encrypted_content"])
        self.assertFalse(requests[0]["store"])

    def test_export_round_trip_and_cli_need_no_api(self):
        run = create_run(self.root, self.settings, "verify")
        update_run(run, selected_claim_ids=["claim_0002", "claim_0001"])
        for cid, verified in [("claim_0001", True), ("claim_0002", False)]:
            save_result(run, {**CLAIM, "claim_id": cid, "claim_text": 'Quotes, "commas"\nand newlines'},
                        Decision(**decision(verified)))
        expected = export_results(run).read_bytes()
        with patch("app.cli.Settings.from_env", side_effect=AssertionError("Export read API config")), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["export", "--run-dir", str(run)]), 0)
        self.assertEqual(expected, (run / "results.csv").read_bytes())
        with (run / "results.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["claim_id"] for r in rows], ["claim_0002", "claim_0001"])
        for row in rows:
            row["verified"] = json.loads(row["verified"])
            row["evidence"] = json.loads(row["evidence"])
            self.assertEqual(row, json.loads((run / "verification" / f"{row['claim_id']}.json").read_text()))

    def test_verify_cli_selects_sample_preserves_ids_and_never_ingests(self):
        prepared = self.root / "papers.csv"
        with prepared.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["paper_id", "Paper Title", "Abstract"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.papers)
        claims = self.root / "claims.csv"
        with claims.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["run_id", "verifiable", *CLAIM])
            writer.writeheader()
            for i in range(1, 4):
                writer.writerow({**CLAIM, "claim_id": f"claim_{i:04d}", "paper_ids": '["A7K2"]',
                                 "run_id": "source_run", "verifiable": "false" if i == 1 else "true"})
        client = Client([response(call()), response(final=decision())])
        with patch("app.cli.Settings.from_env", return_value=Settings(api_key="test")), \
                patch("app.rag.search.open_collection", return_value=self.collection), \
                patch("app.rag.ingest.prepare_rag", side_effect=AssertionError("Verification must not ingest")), \
                patch("openai.OpenAI", return_value=nullcontext(client)), redirect_stdout(io.StringIO()):
            code = main(["verify", "--claims", str(claims), "--prepared", str(prepared),
                         "--claim-limit", "1", "--runs-dir", str(self.root / "runs")])
        self.assertEqual(code, 0)
        record_path = next((self.root / "runs").glob("*/run.json"))
        record = json.loads(record_path.read_text())
        self.assertEqual(record["source_run_id"], "source_run")
        self.assertEqual(record["selected_claim_ids"], ["claim_0002"])
        self.assertEqual(record["skipped_non_verifiable"], 1)
        self.assertEqual(record["unselected_verifiable"], 1)
        self.assertEqual(record["status"], "verification_completed")
        self.assertTrue((record_path.parent / "verification/claim_0002.json").exists())


if __name__ == "__main__":
    unittest.main()
