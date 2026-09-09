"""Offline tests. Model responses are mocked; no API credentials are needed."""

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import httpx
from openai import OpenAI

from app.claim_extraction.extract import extract_claims, paper_catalog, section_input
from app.claim_extraction.sections import split_report
from app.claim_extraction.storage import ClaimStore, SectionClaims, validate_claims
from app.claim_verification.verify import load_verifiable_claims
from app.config import ROOT, Settings
from main import main
from app.results.runs import create_run, update_run
from app.paper_preparation.prepare import generate_paper_id, prepare_papers, read_papers, valid_paper_id

REPORT = """# Sample report
## Summary
Users stopped after six months [20].
## Findings
### Numeric findings
In adults, Device X lowered HbA1c by 0.5 percentage points and increased time in range by 30 minutes after 12 weeks [1], [2].
Monitoring requires charging.
## References
[20] Wearables in Medicine.
"""
CATALOG = [{"paper_id": "A7K2", "title": "Wearables in Medicine"}]


def claim(**changes):
    result = dict(
        claim_text="Users stopped after six months.",
        source_text="Users stopped after six months [20].",
        verifiable=True,
        reference_numbers=[20], paper_ids=["A7K2"],
    )
    result.update(changes)
    return result


def response(claims=None, *, text=None, status="completed"):
    return SimpleNamespace(
        output_text=json.dumps({"claims": claims or []}) if text is None else text,
        status=status,
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = self
        self.queue = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))
        return next(self.queue)


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class PaperTests(TemporaryTest):
    def test_real_collection_preserved_and_ids_stable(self):
        source = ROOT / "input_data/papers.csv"
        original_bytes = source.read_bytes()
        destination = self.root / "papers.csv"
        first = prepare_papers(source, destination)
        saved_bytes = destination.read_bytes()
        self.assertEqual(first, prepare_papers(source, destination))
        self.assertEqual(162, len(first))
        self.assertEqual(saved_bytes, destination.read_bytes())
        self.assertEqual(original_bytes, source.read_bytes())
        ids = [row["paper_id"] for row in first]
        self.assertTrue(all(valid_paper_id(pid) for pid in ids))
        self.assertEqual(len(ids), len(set(ids)))
        fields, original = read_papers(source)
        self.assertEqual(original, [{key: row[key] for key in fields} for row in first])

    def test_collisions_and_numeric_ids_are_retried(self):
        with patch("app.paper_preparation.prepare.secrets.choice", side_effect=list("1234AAAAA7K2B2C3")):
            self.assertEqual("B2C3", generate_paper_id({"A7K2"}))

    def test_changed_input_does_not_silently_reuse_ids(self):
        source = self.root / "source.csv"
        source.write_text('Paper Title,Abstract\nOne,"Text, with\na newline"\n')
        destination = self.root / "papers.csv"
        prepare_papers(source, destination)
        before = destination.read_bytes()
        source.write_text("Paper Title,Abstract\nTwo,changed\n")
        with self.assertRaisesRegex(ValueError, "Input papers changed"):
            prepare_papers(source, destination)
        self.assertEqual(before, destination.read_bytes())


class SectionTests(unittest.TestCase):
    def test_real_report_has_10_intact_content_sections(self):
        text = (ROOT / "input_data/report.md").read_text()
        report = split_report(text)
        self.assertEqual(10, len(report.sections))
        self.assertEqual("1. Introduction", report.sections[0].heading)
        self.assertFalse(any(s.heading in {"Executive Summary", "References", "Table of Contents"} for s in report.sections))
        self.assertTrue(report.bibliography.startswith("## References"))
        outcomes = next(s for s in report.sections if s.heading.startswith("5."))
        expected = text[text.index("## 5."):text.index("## 6.")]
        self.assertEqual(expected, outcomes.text)
        self.assertIn("###", outcomes.text)

    def test_preamble_deeper_headings_fences_and_duplicate_names(self):
        text = "# Title\nPreamble fact.\n## Same\nA\n### Child\nB\n```md\n## Fake\n```\n## Same\nC\n## Bibliography\nBook\n## Table of Contents\nLinks\n"
        report = split_report(text)
        self.assertEqual(["Preamble", "Same", "Same"], [s.heading for s in report.sections])
        self.assertEqual(3, len({s.section_id for s in report.sections}))
        self.assertIn("## Fake", report.sections[1].text)
        self.assertIn("### Child", report.sections[1].text)
        self.assertIn("Book", report.bibliography)
        self.assertNotIn("Links", "".join(s.text for s in report.sections))

    def test_no_h2_is_one_section(self):
        text = "# Title\nA fact.\n### Details\nMore facts.\n"
        report = split_report(text)
        self.assertEqual(1, len(report.sections))
        self.assertEqual(text, report.sections[0].text)

    def test_catalog_excludes_every_other_csv_field(self):
        papers = [{"paper_id": "A7K2", "Paper Title": "Wearables in Medicine", "Abstract": "DO NOT SEND", "DOI": "DO NOT SEND"}]
        catalog = paper_catalog(papers)
        self.assertEqual(CATALOG, catalog)
        report = split_report(REPORT)
        request = section_input(report, report.sections[0], catalog)
        self.assertNotIn("DO NOT SEND", request)
        self.assertNotIn("Numeric findings", request)
        self.assertIn("[20] Wearables in Medicine", request)


class StorageTests(TemporaryTest):
    def test_non_verifiable_claims_are_saved_but_skipped_before_sample_limit(self):
        section = split_report("## Findings\nWearables are the future of healthcare.\nUsers stopped after six months [20].\nMonitoring requires charging.\n").sections[0]
        batch = SectionClaims.model_validate({"claims": [
            claim(claim_text="Wearables are the future of healthcare.", source_text="Wearables are the future of healthcare.", verifiable=False, reference_numbers=[], paper_ids=[]),
            claim(reference_numbers=[], paper_ids=[]),
            claim(claim_text="Monitoring requires charging.", source_text="Monitoring requires charging.", reference_numbers=[], paper_ids=[]),
        ]})
        validate_claims(batch, section, {"A7K2"})
        store = ClaimStore(self.root)
        store.save_section(section, batch)
        saved = json.loads((store.json_dir / "section_0001.json").read_text())["claims"]
        self.assertIs(False, saved[0]["verifiable"])
        self.assertEqual(3, store.count)
        selected = load_verifiable_claims(store.csv_path, limit=1)
        self.assertEqual(["claim_0002"], [c["claim_id"] for c in selected])
        self.assertIs(True, selected[0]["verifiable"])

    def test_older_csv_requires_fresh_extraction(self):
        path = self.root / "old.csv"
        path.write_text("claim_id,claim_text\nclaim_0001,Example\n")
        with self.assertRaisesRegex(ValueError, "missing verifiable"):
            load_verifiable_claims(path)

    def test_section_saves_are_duplicate_safe_and_empty_results_are_saved(self):
        sections = split_report("## Same\nUsers stopped after six months [20].\n## Same\nUsers stopped after six months [20].\n## Empty\n").sections
        store = ClaimStore(self.root)
        batch = SectionClaims.model_validate({"claims": [claim(), claim()]})
        validate_claims(batch, sections[0], {"A7K2"})
        store.save_section(sections[0], batch)
        store.save_section(sections[0], batch)
        store.save_section(sections[1], batch)
        store.save_section(sections[2], SectionClaims(claims=[]))
        self.assertEqual(2, store.count)
        with store.csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(["claim_0001", "claim_0002"], [r["claim_id"] for r in rows])
        self.assertNotEqual(rows[0]["section_id"], rows[1]["section_id"])
        self.assertEqual(["A7K2"], json.loads(rows[0]["paper_ids"]))
        self.assertEqual("true", rows[0]["verifiable"])
        self.assertEqual([], json.loads((store.json_dir / "section_0003.json").read_text())["claims"])

    def test_numeric_compound_and_uncited_fixture_preserves_qualifiers(self):
        section = split_report(REPORT).sections[1]
        source = section.text.splitlines()[2]
        claims = [claim(
            claim_text=text, source_text=source, reference_numbers=[1, 2], paper_ids=["A7K2", "B2C3"],
        ) for text in [
            "In adults, Device X lowered HbA1c by 0.5 percentage points after 12 weeks.",
            "In adults, Device X increased time in range by 30 minutes after 12 weeks.",
        ]]
        claims.append(claim(claim_text="Monitoring requires charging.", source_text="Monitoring requires charging.", reference_numbers=[], paper_ids=[]))
        batch = SectionClaims.model_validate({"claims": claims})
        validate_claims(batch, section, {"A7K2", "B2C3"})
        store = ClaimStore(self.root)
        store.save_section(section, batch)
        saved = json.loads((store.json_dir / f"{section.section_id}.json").read_text())["claims"]
        self.assertIn("0.5 percentage points after 12 weeks", saved[0]["claim_text"])
        self.assertEqual([], saved[2]["paper_ids"])


class ExtractionTests(TemporaryTest):
    def run_extraction(self, responses, report=None, settings=None):
        store = ClaimStore(self.root)
        client = FakeClient(responses)
        result = extract_claims(client, settings or Settings(), report or split_report(REPORT), CATALOG, store)
        return result, store, client

    def test_one_fresh_json_request_per_section(self):
        result, store, client = self.run_extraction([response([claim()]), response()])
        self.assertEqual(2, len(result["completed_sections"]))
        self.assertEqual([], result["failed_sections"])
        self.assertEqual(1, store.count)
        self.assertTrue(all(len(r["input"]) == 1 for r in client.requests))
        self.assertTrue(all("tools" not in r for r in client.requests))
        self.assertEqual("json_schema", client.requests[0]["text"]["format"]["type"])

    def test_invalid_paper_id_is_corrected_before_any_save(self):
        result, store, client = self.run_extraction([response([claim(paper_ids=["0020"])]), response([claim()]), response()])
        self.assertEqual(1, store.count)
        self.assertEqual([], result["failed_sections"])
        self.assertEqual(3, len(client.requests))
        self.assertIn("paper_ids must come", client.requests[1]["input"][-1]["content"])
        self.assertEqual(1, len(client.requests[2]["input"]))

    def test_bad_responses_fail_section_without_partial_writes(self):
        invalid = [
            response(text="{bad json"), response(text=""), response(status="incomplete"),
            response([claim(reference_numbers=[True])]),
            response([claim(source_text="Monitoring requires charging.")]),
            response([claim(), claim(paper_ids=["ZZ99"])]),
            response([claim(claim_id="model-assigned-id")]),
            response([claim(verifiable="true")]),
            response([claim(verifiable=1)]),
        ]
        for index, bad in enumerate(invalid):
            with self.subTest(index=index):
                store = ClaimStore(self.root / str(index))
                client = FakeClient([bad, bad, response()])
                result = extract_claims(client, Settings(), split_report(REPORT), CATALOG, store)
                self.assertEqual(1, len(result["failed_sections"]))
                self.assertEqual(["section_0002"], result["completed_sections"])
                self.assertEqual(0, store.count)
                self.assertFalse((store.json_dir / "section_0001.json").exists())
                self.assertEqual(3, len(client.requests))

    def test_real_sdk_round_trip_uses_structured_json_without_network(self):
        requests = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append(payload)
            text = json.dumps({"claims": [claim()] if len(requests) == 1 else []})
            return httpx.Response(200, json={
                "id": f"resp_{len(requests)}", "object": "response", "created_at": 0,
                "model": "test-model", "status": "completed",
                "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            })

        store = ClaimStore(self.root)
        with OpenAI(api_key="fake-test-key", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            result = extract_claims(client, Settings(extraction_model="test-model", verification_model="different-model"), split_report(REPORT), CATALOG, store)
        self.assertEqual(1, result["claim_count"])
        self.assertEqual(2, len(requests))
        self.assertTrue(all(request["model"] == "test-model" for request in requests))
        self.assertTrue(requests[0]["text"]["format"]["strict"])
        self.assertNotIn("max_output_tokens", requests[0])
        self.assertEqual(CATALOG, json.loads(requests[0]["input"][0]["content"])["paper_catalog"])
        self.assertEqual("disabled", requests[0]["truncation"])


class RunTests(TemporaryTest):
    def test_launcher_and_default_paths_from_another_directory(self):
        launcher = [sys.executable, str(ROOT / "main.py")]
        help_result = subprocess.run(launcher + ["--help"], cwd=self.root,
                                     capture_output=True, text=True, check=True)
        self.assertIn("prepare-rag", help_result.stdout)
        prepared = self.root / "papers.csv"
        for command, extra, status in [
            ("prepare", [], "prepared"),
            ("extract", ["--dry-run"], "ready"),
            ("prepare-rag", ["--dry-run"], "ready"),
        ]:
            with self.subTest(command=command):
                runs = self.root / command
                subprocess.run(launcher + [command, "--prepared", str(prepared),
                                           "--runs-dir", str(runs), *extra],
                               cwd=self.root, capture_output=True, text=True, check=True)
                record = json.loads(next(runs.glob("*/run.json")).read_text())
                self.assertEqual(status, record["status"])
                if command == "extract":
                    self.assertEqual(10, record["available_section_count"])

    def test_cli_records_completed_and_unfinished_sections(self):
        source = self.root / "source.csv"
        source.write_text("Paper Title\nWearables in Medicine\n")
        report = self.root / "report.md"
        report.write_text(REPORT)
        for case, outputs, expected_status, expected_count in [
            ("complete", [response([claim(paper_ids=[])]), response()], "extraction_completed", 1),
            ("partial", [response([claim(paper_ids=[])]), response(text="bad"), response(text="bad")], "incomplete", 1),
            ("sample", [response([claim(paper_ids=[])] )], "extraction_completed", 1),
        ]:
            with self.subTest(case=case):
                client = FakeClient(outputs)
                args = ["extract", "--papers", str(source), "--report", str(report),
                        "--prepared", str(self.root / "prepared.csv"), "--runs-dir", str(self.root / case)]
                if case == "sample":
                    args.extend(["--section-limit", "1"])
                with patch("app.cli.Settings.from_env", return_value=Settings(api_key="test-key", extraction_model="test-model")), \
                        patch("openai.OpenAI", return_value=nullcontext(client)), \
                        redirect_stdout(io.StringIO()):
                    exit_code = main(args)
                run_path = next((self.root / case).glob("*/run.json"))
                record = json.loads(run_path.read_text())
                self.assertEqual(expected_status, record["status"])
                self.assertEqual(expected_count, record["claim_count"])
                self.assertEqual(1 if case == "partial" else 0, exit_code)
                self.assertEqual(2, record["available_section_count"])
                self.assertEqual(1 if case == "sample" else 2, len(record["sections"]))
                with (run_path.parent / "claims.csv").open(newline="") as stream:
                    self.assertEqual(expected_count, len(list(csv.DictReader(stream))))

    def test_runs_are_unique_and_settings_exclude_secrets(self):
        settings = Settings(api_key="secret-must-not-appear", cohere_key="cohere-secret-must-not-appear")
        first = create_run(self.root, settings, "extract")
        before = (first / "run.json").read_text()
        second = create_run(self.root, settings, "extract")
        self.assertNotEqual(UUID(first.name), UUID(second.name))
        self.assertTrue((first / "extraction").is_dir())
        self.assertTrue((first / "verification").is_dir())
        self.assertEqual(before, (first / "run.json").read_text())
        self.assertNotIn(settings.api_key, before)
        self.assertNotIn(settings.api_key, repr(settings))
        self.assertNotIn(settings.cohere_key, before)
        self.assertNotIn(settings.cohere_key, repr(settings))
        update_run(first, status="failed", error="Example failure")
        self.assertEqual("failed", json.loads((first / "run.json").read_text())["status"])

    def test_invalid_config_is_clear_and_does_not_echo_values(self):
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY, EXTRACTION_MODEL"):
            Settings(extraction_model="").require_openai()
        with patch("app.config.load_dotenv"), patch.dict("os.environ", {"MAX_TOOL_CALLS": "secret-value"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MAX_TOOL_CALLS must be a positive integer"):
                Settings.from_env()

    def test_models_are_independent_and_default_to_terra(self):
        with patch("app.config.load_dotenv"), patch.dict("os.environ", {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual("gpt-5.6-terra", settings.extraction_model)
        self.assertEqual("gpt-5.6-terra", settings.verification_model)
        self.assertEqual(20, settings.verification_claim_limit)
        with patch("app.config.load_dotenv"), patch.dict("os.environ", {
            "EXTRACTION_MODEL": "extract-model", "VERIFICATION_MODEL": "verify-model",
            "VERIFICATION_CLAIM_LIMIT": "5",
        }, clear=True):
            settings = Settings.from_env()
        self.assertEqual("extract-model", settings.extraction_model)
        self.assertEqual("verify-model", settings.verification_model)
        self.assertEqual(5, settings.verification_claim_limit)
        Settings(api_key="test", verification_model="").require_openai("extraction")
        with self.assertRaisesRegex(ValueError, "VERIFICATION_MODEL"):
            Settings(api_key="test", verification_model="").require_openai("verification")
        with patch("app.config.load_dotenv"), patch.dict("os.environ", {"VERIFICATION_CLAIM_LIMIT": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "VERIFICATION_CLAIM_LIMIT must be a positive integer"):
                Settings.from_env()

    def test_missing_credentials_record_a_failed_run(self):
        with patch("app.cli.Settings.from_env", return_value=Settings()), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["extract", "--prepared", str(self.root / "papers.csv"), "--runs-dir", str(self.root / "runs")])
        self.assertEqual(1, code)
        record = json.loads(next((self.root / "runs").glob("*/run.json")).read_text())
        self.assertEqual("failed", record["status"])
        self.assertIn("OPENAI_API_KEY", record["error"])


if __name__ == "__main__":
    unittest.main()
