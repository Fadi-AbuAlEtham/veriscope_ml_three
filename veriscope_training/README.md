# VeriScope Training

Server-side phishing model training and evaluation pipeline for VeriScope.

## Running VeriScope Data Acquisition on a Remote Server

This project now supports a server-first raw-data acquisition workflow. Dataset fetching and snapshot preparation are designed to run directly on a Linux server or cloud VM and write into `data/raw/<dataset>/`.

Recommended remote workflow:

1. SSH into the server.
2. Clone the repository on the server.
3. Create and activate a virtual environment.
4. Install dependencies from `requirements.txt`.
5. Run raw dataset acquisition commands on the server.
6. Validate raw datasets.
7. Build processed datasets.
8. Train models.
9. Evaluate and compare runs.

Example:

```bash
ssh user@your-server
git clone <repo-url>
cd veriscope_training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 -m veriscope_training.cli.main --project-root . list-fetch-sources --json
python3 -m veriscope_training.cli.main --project-root . fetch-dataset phreshphish --json
python3 -m veriscope_training.cli.main --project-root . validate-raw-dataset phreshphish --json
python3 -m veriscope_training.cli.main --project-root . build-dataset --json
python3 -m veriscope_training.cli.main --project-root . evaluate-all --group full_suite --json
```

### Raw Acquisition Modes

- Auto-fetchable on server:
  - `phreshphish`
  - `openphish`
  - `phishtank`
  - `uci_phishing`
  - `oscar_aux` with Hugging Face access
- Streaming-capable preview support:
  - `phreshphish`
  - `oscar_aux`
- Manual server snapshot only:
  - `mendeley`
- Auth / approval dependent:
  - `openphish` academic or premium variants
  - `phishtank` app-key-enhanced bulk fetch
  - `oscar_aux` gated Hugging Face access may require `HF_TOKEN`

Nothing in this workflow requires downloading datasets onto your laptop first.

## Phase 5 Experiment Evaluation and VeriScope Integration Outputs

The project now includes:

- configuration-driven experiment suites
- aggregate comparison across URL, webpage, transformer, and tabular tracks
- threshold calibration and risk-bucket mapping
- structured error analysis over saved predictions
- data-driven model recommendation outputs
- integration-ready config exports for VeriScope server-side services

All ML/NLP inference remains server-side only.

### Experiment Orchestration

Phase 5 adds named experiment groups in `configs/experiments.yaml`:

- `url_baselines`
- `webpage_baselines`
- `transformer_comparison`
- `tabular_baselines`
- `baseline_suite`
- `full_suite`

Each group can:

- run multiple track-specific experiments reproducibly
- skip already completed runs when configured
- continue after dependency or data failures when configured
- emit experiment manifests, comparison tables, recommendations, calibration outputs, and integration configs

### Evaluation Aggregation and Reporting

Saved training runs are compared from their existing artifacts:

- `run_summary.json`
- `reports/metrics.json`
- `reports/*_predictions.jsonl`
- `config_snapshot.json`
- `package_versions.json`

Comparison outputs include:

- JSON and CSV comparison tables
- Markdown comparison reports
- per-metric comparison bar charts
- recommendation summaries
- calibration summaries and plots
- error analysis summaries

Primary comparison emphasis remains:

- precision
- recall
- f1
- pr_auc
- false_positive_rate
- false_negative_rate

### Threshold Calibration and Risk Mapping

Threshold calibration operates on saved prediction files and supports:

- `maximize_f1`
- `recall_under_precision_floor`
- `min_false_positives_under_recall`

Risk mapping produces:

- binary phishing threshold
- `low` / `medium` / `high` risk levels
- saved threshold rationale and threshold-performance tables
- exportable per-model threshold and risk-mapping configs

### VeriScope Integration-Ready Outputs

Phase 5 exports:

- model recommendation config
- threshold config
- risk-mapping config
- score interpretation config
- prediction schema example for future server API integration

The exported prediction schema is designed for future VeriScope fusion with:

- client-side heuristics
- server-side model scores
- structural webpage indicators
- future reason-code / evidence bundles

## Phase 4 Training and Adaptive Foundations

The project now includes:

- reproducible split strategies for processed datasets
- URL baseline training
- webpage text baseline training
- multilingual transformer training for webpage/text data
- tabular baseline training
- saved run artifacts, split manifests, metrics, and predictions
- safe adaptive-learning foundations for feedback logging, drift monitoring, retraining-candidate export, and heuristic proposal generation

### Multilingual Transformer Scope

Transformer training is server-side only.

- Primary multilingual model: `xlm-roberta-base`
- Comparison baseline: `bert-base-multilingual-cased`

MiniLM is not part of the required Phase 4 transformer scope.

Transformers are used only for webpage/text input. URL-only and tabular tracks remain separate classical pipelines.

### Supported Training Tracks

- URL track
  - `tfidf_logreg`
  - `tfidf_linear_svm`
  - `handcrafted_boosting`
- Webpage text track
  - `text_tfidf_logreg`
  - `text_tfidf_linear_svm`
- Multilingual transformer track
- `xlmr_sequence_classifier`
- `mbert_sequence_classifier`
  CLI aliases: `xlmr`, `mbert`
- Tabular track
  - `logistic_regression`
  - `random_forest`
  - `gradient_boosting_optional`

### Split Strategies

Implemented split strategies:

- `random_stratified`
- `source_aware`
- `domain_aware`
- `time_aware`
- `predefined_source`

Split manifests also record grouping fields, timestamp coverage, and fallback reasons when a requested strategy has to degrade safely, such as a very small dataset that cannot support stratified or temporal holdouts.

Each training run saves:

- `split_manifest.json`
- `split_assignments.jsonl`
- metrics and confusion matrix plot
- label metadata
- model/vectorizer/tokenizer artifacts
- package versions
- run summary

### Adaptive-Learning Foundation

The adaptive layer is intentionally review-driven and non-autonomous.

Implemented components:

- feedback record schema for reviewed outcomes
- drift comparison utilities for label/language/source/TLD/token/score shifts
- retraining-candidate export from uncertain or reviewed prediction outcomes
- heuristic proposal generation with evidence and review-ready metadata

Heuristic proposals are never auto-activated. They are emitted with `status = proposed` and intended for human review dashboards or analyst workflows.

## Phase 3 Preprocessing and Unified Dataset Construction

The project now supports:

- raw adapter ingestion from the six dataset sources
- canonical record normalization
- safe URL normalization
- HTML parsing and visible-text extraction
- light multilingual-safe text normalization
- centralized label normalization
- deterministic first-seen deduplication with saved duplicate reports
- separate processed unified outputs for URL, webpage, tabular, and auxiliary-text views

Processed outputs are written under `data/processed/` and manifests/reports under `data/manifests/`.

### Canonical Processed Record

Each raw adapter record is transformed into a processed canonical record that preserves:

- `sample_id`
- `source_dataset`
- `source_original_id`
- `source_split`
- `source_label`
- `normalized_label`
- `original_url`
- `normalized_url`
- `raw_html`
- `extracted_text`
- `normalized_text`
- modality flags
- URL/HTML/text derived metadata
- original adapter metadata

This is not yet final ML feature engineering. It is the cleaned canonical layer used by later training phases.

### Unified Output Views

- `unified_url_dataset`
  - records with usable normalized URL information
- `unified_webpage_dataset`
  - supervised records with HTML and/or extracted webpage text
- `unified_tabular_dataset`
  - structured records such as UCI phishing features
- `unified_auxiliary_text_dataset`
  - auxiliary unlabeled text such as OSCAR

### Deduplication Strategy

Deduplication is deterministic and currently uses a `first_seen` policy. Matching keys include:

- source-specific original ID when present
- exact original URL
- normalized URL
- raw HTML hash
- normalized text hash

Removed duplicates are not discarded silently:

- summary report: `data/manifests/build_dataset.dedupe_report.json`
- duplicate events: `data/manifests/build_dataset.duplicates.jsonl`

This phase implements exact dedupe and a safe foundation for later near-duplicate work. It does not attempt fuzzy near-duplicate removal yet.

## Phase 2 Dataset Adapters

The project now includes source-specific adapters for:

- `phreshphish`
- `openphish`
- `phishtank`
- `uci_phishing`
- `mendeley`
- `oscar_aux`

These adapters normalize each source into the shared `DatasetRecord` schema while preserving provenance, raw source metadata, and modality honesty.

## Dataset Acquisition Notes

### PhreshPhish

- Preferred remote source: Hugging Face dataset `phreshphish/phreshphish`
- Supported local snapshot formats: `parquet`, `jsonl`, `json`, `csv`, `tsv`
- Expected fields when local: `url`, `html`, `label`, optional `split`, `date`, `lang`, `sha256`
- Adapter behavior:
  - keeps webpage-oriented fields intact
  - maps `phish/phishing` to `1`
  - maps `benign/legitimate` to `0`
  - leaves extracted text empty unless the snapshot already contains it

### OpenPhish

- Supported local snapshot formats: `txt`, `csv`, `tsv`, `jsonl`, `json`, `parquet`
- Common community snapshot format: one phishing URL per line in a text file
- Adapter behavior:
  - treats the source as phishing-only
  - sets `normalized_label = 1`
  - preserves timestamps if present in structured snapshots
  - does not fabricate benign examples

### PhishTank

- Supported local snapshot formats: `csv`, `xml`, `json`, including `.gz` and `.bz2`
- Official downloadable feeds commonly use `online-valid.csv`, `online-valid.xml`, or `online-valid.json`
- Preserved metadata includes `phish_id`, `phish_detail_url`, `submission_time`, `verification_time`, `online`, `target`, and network detail blocks when present
- Adapter behavior:
  - maps verified phishing-confirmed entries to `1`
  - leaves non-confirmed or ambiguous rows unlabeled instead of inventing negatives

### UCI Phishing Websites

- Supported local snapshot formats: `arff`, `csv`, `tsv`, `jsonl`, `json`, `parquet`
- Recommended local file: UCI `Training Dataset.arff`
- Adapter behavior:
  - treats the source as tabular only
  - stores all non-label columns under `tabular_features`
  - uses the classic mapping for the UCI phishing websites dataset:
    - `-1 -> phishing -> 1`
    - `1 -> benign -> 0`
    - `0 -> suspicious/uncertain -> null`

### Mendeley

- Supported local snapshot formats: `csv`, `tsv`, `jsonl`, `json`, `parquet`, `html`, `htm`, `txt`
- Snapshot structures vary by mirror and paper supplement
- Recommended local structured fields when available: `url`, `html`, `text/content`, `label`, optional `split`, `language`, `timestamp`
- Raw file fallback:
  - `.html/.htm` files are ingested as HTML samples
  - `.txt` files are ingested as text samples
  - labels can be inferred from parent directory names like `phishing/` or `benign/`
- Numeric labels are intentionally treated as ambiguous by default unless you override `label_map` in `configs/datasets.yaml`

### OSCAR Auxiliary

- Supported local snapshot formats: `jsonl`, `json`, `parquet`, `csv`, `tsv`, `txt`
- Optional remote source: Hugging Face `oscar-corpus/oscar`
- Adapter behavior:
  - ingests text for auxiliary multilingual support only
  - keeps `normalized_label = null`
  - marks records as auxiliary and unlabeled in metadata
  - should not be used as supervised phishing training data

## Adapter Inspection CLI

Examples:

```bash
veriscope-training list-datasets
veriscope-training list-fetch-sources --json
veriscope-training inspect-fetch-config phreshphish --json
veriscope-training fetch-dataset phreshphish --json
veriscope-training fetch-dataset openphish --json
veriscope-training fetch-dataset phishtank --json
veriscope-training fetch-dataset uci_phishing --json
veriscope-training fetch-dataset mendeley --json
veriscope-training fetch-dataset oscar_aux --json
veriscope-training fetch-all-datasets --json
veriscope-training validate-raw-dataset phreshphish --json
veriscope-training list-models --json
veriscope-training inspect-source phreshphish --json
veriscope-training inspect-adapter phishtank --json
veriscope-training preview-records openphish --limit 5 --json
veriscope-training build-dataset --json
veriscope-training build-dataset --force-rebuild --json
veriscope-training build-dataset --no-skip-completed --output-format parquet --json
veriscope-training build-dataset --source phreshphish --view webpage --json
veriscope-training preview-processed --source openphish --view url --limit 5 --json
veriscope-training show-manifest unified_url_dataset --json
veriscope-training show-dedupe-report --events --limit 10 --json
veriscope-training train-model --track url --model tfidf_logreg --json
veriscope-training train-model --track webpage --model text_tfidf_linear_svm --json
veriscope-training train-model --track transformer --model xlmr_sequence_classifier --json
veriscope-training train-model --track transformer --model mbert_sequence_classifier --json
# shorthand aliases are also accepted:
veriscope-training train-model --track webpage --model xlmr --json
veriscope-training train-model --track webpage --model mbert --json
veriscope-training train-model --track tabular --model random_forest --json
veriscope-training train-all-baselines --include-transformers --json
veriscope-training show-training-run outputs/training/url/tfidf_logreg/<run_name> --json
veriscope-training evaluate-all --group full_suite --json
veriscope-training compare-runs --json
veriscope-training calibrate-thresholds --run-dir outputs/training/webpage_transformer/xlmr_sequence_classifier/<run_name> --json
veriscope-training error-analysis --run-dir outputs/training/webpage_transformer/xlmr_sequence_classifier/<run_name> --json
veriscope-training export-integration-configs --json
veriscope-training export-retraining-candidates --predictions outputs/training/url/tfidf_logreg/<run_name>/reports/test_predictions.jsonl --json
veriscope-training generate-heuristic-proposals --processed data/processed/unified_webpage_dataset.parquet --json
veriscope-training show-drift-report --reference data/processed/unified_webpage_dataset.parquet --current outputs/training/webpage/text_tfidf_logreg/<run_name>/reports/test_predictions.jsonl --json
python3 scripts/run_build_dataset.py
python3 scripts/run_fetch_dataset.py <dataset_name>
python3 scripts/run_fetch_all_datasets.py
python3 scripts/run_evaluate_all.py
python3 scripts/run_compare_runs.py
```

## Assumptions

- OpenPhish and PhishTank are phishing-heavy feeds and are not used to synthesize benign labels.
- PhreshPhish may be loaded from Hugging Face when the optional `datasets` package is installed and local snapshots are absent.
- Some OSCAR variants are gated or temporarily unavailable remotely, so local snapshots are the safest operational path.
- Remote acquisition writes directly into `data/raw/<dataset>/` on the server.
- `HF_TOKEN`, `OPENPHISH_FETCH_URL`, `OPENPHISH_USERNAME`, `OPENPHISH_PASSWORD`, `OPENPHISH_TOKEN`, `PHISHTANK_APP_KEY`, and `PHISHTANK_USER_AGENT` are server-side environment variables only and are never hardcoded.
- OpenPhish community-mode fetching uses the official free feed linked from the OpenPhish phishing feeds page.
- PhishTank auto-fetch uses the official developer bulk-download endpoints and benefits from a registered app key when available.
- UCI phishing acquisition uses the official UCI dataset through `ucimlrepo` and materializes a local CSV snapshot for the existing tabular adapter.
- Mendeley acquisition remains manual-on-server because mirror access and packaging vary.
- Mendeley mirrors vary, so the adapter is intentionally permissive and preserves raw source fields in metadata rather than flattening them aggressively.
- Canonical text normalization is intentionally light. It normalizes whitespace and control characters but does not strip punctuation, numbers, URLs, or urgency language by default.
- URL normalization lowercases only scheme and hostname while preserving path/query evidence.
- Processed outputs default to Parquet with compression. JSONL is available only as an explicit fallback.
- Default processed-storage policy drops raw HTML from the materialized webpage view to reduce disk usage while preserving extracted text, normalized text, hashes, and provenance metadata.
- Processed builds skip already materialized views by default and write lightweight duplicate-event samples instead of unbounded duplicate logs.
- XLM-R is the primary multilingual transformer because it is a stronger multilingual encoder baseline for server-side text classification; mBERT is included as a comparison baseline, not the default choice.
- Transformer training requires `transformers`, `datasets`, `torch`, and `accelerate`. If these dependencies are unavailable, transformer commands fail explicitly instead of silently falling back.
- Transformer text training uses `normalized_text` first and falls back to `extracted_text`. Raw HTML is not used as default training text.
- Gradient boosting for tabular data uses `xgboost` or `lightgbm` only if installed.
- Phase 5 comparison, calibration, and recommendation outputs depend on saved training runs and prediction files from Phase 4.
- Threshold calibration is transparent and configuration-driven; it is not an auto-deploy mechanism.
- Integration exports are server-side configuration artifacts only. They do not move ML/NLP inference into the client.
- Adaptive outputs are proposal/report artifacts only. They do not change production models or heuristic rules automatically.

## Limitations

- Real processed-output validation still depends on actual snapshots being present under `data/raw/`.
- Deduplication is exact-key based in this phase and does not yet perform fuzzy text or DOM similarity matching.
- Visible text extraction is safe and pragmatic, not a full browser-rendering simulation.
- Training commands require processed datasets to exist first.
- Transformer runs can be resource-intensive and are intended for remote/server/cloud execution, not client-side inference.
- Phase 5 reports are only as strong as the saved run coverage. If real processed data or completed runs are absent, comparison and calibration commands fail clearly rather than fabricating results.
- Some acquisition paths remain approval- or license-dependent. When automation is not possible, the project prepares server-side target directories and instructions rather than pretending to fetch inaccessible data.
