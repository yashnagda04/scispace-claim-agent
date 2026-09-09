# SciSpace claim evaluation

A local Python application that extracts claims from a research report and checks
whether the supplied papers support them. Results contain a `verified` boolean,
evidence, and a short reason. `false` means support was not established and the
claim needs review; it does not necessarily mean the claim is factually false.

## Folder structure

```text
app/                       Application code and prompts
├── cli.py                 Command handlers
├── config.py              Configuration and defaults
├── paper_preparation/     Persistent paper IDs
├── claim_extraction/      Extract claims from report sections
├── rag/                   PDF processing, embeddings, search, and reranking
├── claim_verification/    Verify claims using retrieved evidence
└── results/               Run tracking and CSV export
main.py                    Command-line launcher
input_data/                Original report and paper CSV
data/                      Prepared papers, PDFs, text cache, and Chroma index
runs/                      Generated extraction and verification runs
report/                    Architecture, full extraction, and sample results
output/                    Presentation files
tests/                     Offline tests
```

## Setup

Use Python 3.11 or newer. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for extraction, embeddings, and verification.
Set `COHERE_KEY` to enable reranking: retrieve up to 100 candidates and return
the best 20 passages. Without it, search returns up to 20 vector matches.

Configure model names in `.env`; [.env.example](.env.example) lists the settings.
Extraction and verification default to `gpt-5.6-terra`; embeddings use
`text-embedding-3-small`. The default verification sample is 20 verifiable
claims, with a separate budget of 20 retrieval-tool calls per claim.

## Run

The default inputs are `input_data/report.md` and `input_data/papers.csv`.

```sh
# Assign or reuse stable paper IDs.
python main.py prepare

# Extract claims from all report body sections.
python main.py extract

# Download available PDFs and build or update the local index.
python main.py prepare-rag

# Verify the first five verifiable claims from an extraction run.
python main.py verify --claims runs/EXTRACTION_RUN_ID/claims.csv --claim-limit 5

# Regenerate a verification run's CSV from its saved JSON, without API calls.
python main.py export --run-dir runs/VERIFICATION_RUN_ID
```

Replace `EXTRACTION_RUN_ID` and `VERIFICATION_RUN_ID` with the run IDs printed
by the corresponding commands. Existing PDFs, extracted text, and unchanged
embeddings are reused. Verification uses the existing index.

For a smaller extraction, use `extract --section-limit 2`. Add `--dry-run` to
`extract`, `prepare-rag`, or `verify` to preview work without API calls.
See `python main.py --help` for input paths and other options.

Run the offline tests:

```sh
python -m unittest discover -s tests -v
```

## Outputs and report

Extraction and verification create separate runs under `runs/<run_id>/`:

- Extraction: `claims.csv` and per-section JSON in `extraction/`.
- Verification: `results.csv` and per-claim JSON in `verification/`.
- Run settings, progress, and errors: `run.json`.

The report's output folder contains two CSVs:

- [All claims](report/74275812-e54e-4e5f-874a-613186a73561/claims.csv): 322 extracted claims, including 281 marked verifiable.
- [Results CSV](report/74275812-e54e-4e5f-874a-613186a73561/results.csv): five completed checks, **two verified and three needing review**, with evidence and reasons.

Raw extraction and verification files remain under `runs/`.

Read [the main report](report/assignment-explained.md) for the architecture,
how extraction and verification work, index coverage, and a verified example.
