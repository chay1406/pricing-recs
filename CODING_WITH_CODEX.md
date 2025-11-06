# CODING_WITH_CODEX.md
**How we build this repo using the _Codex – OpenAI’s coding agent_ VS Code extension**

This guide shows how to delegate well-scoped tasks to Codex (OpenAI’s agent in VS Code) to implement our **pricing-recs** plan. It includes setup, guardrails, and ready-to-paste tasks for Phase 1 and Phase 2.

## 0) Prereqs

- Python ≥ 3.10
- VS Code latest
- ChatGPT Plus/Pro/Business (for Codex extension sign-in)
- Repo: `git clone https://github.com/<your-org-or-user>/pricing-recs.git`

## 1) Install & open Codex

1. In VS Code, install **“Codex – OpenAI’s coding agent.”**
2. Sign in with your OpenAI account.
3. Trust the workspace.
4. Use the Command Palette → **Codex: New Task** to run tasks.

## 2) House rules (paste into every task)

```
Constraints:
- Modify only files in this repo and show a short plan first.
- Do not run shell/network commands without asking; print commands for me to run.
- Default to pandas mode; guard Spark imports (don’t break local runs).
- Add/modify unit tests; `pytest -q` must pass.
- Include type hints and docstrings.
- Keep changes minimal and scoped to listed files.
```

## 3) Project conventions

- Python version: 3.10
- Formatting: Black (line length 88)
- Tests: pytest
- Config: per-BU YAML in `configs/`
- Pandas vs Spark: both supported; pandas is default for local dev
- MLflow: log params/metrics; registry name pattern `pricing_recs_<bu_id>`
- Outputs: CSV (pandas) or Delta (spark) paths configured in YAML

## 4) Phase 1 — Orders-only training pipeline

### Task A — IO adapters + orders reader

```
Goal: Implement dual-mode DataFrame adapters and wire OrdersReader.

Files:
- Create `src/pricing/io/adapters.py`
- Edit `src/pricing/io/connectors.py`
- Create tests `tests/unit/test_io_adapters.py`

Details:
- Adapters: PdAdapter and SpAdapter implementing `to_pandas()` and `to_spark(spark)`.
- OrdersReader.read(config): read CSV/Delta from config.input_table; return PdAdapter (pandas) or SpAdapter (spark).
- Guard Spark imports with a friendly error if Spark is unavailable.

Acceptance:
- `pytest -q` passes; round-trip test skips Spark if not installed.
```

### Task B — FeatureBuilder v1 (orders-only)

```
Goal: Implement `FeatureBuilder.fit/transform` for orders-only.

Files:
- `src/pricing/features/feature_builder.py`
- `tests/unit/test_features.py`

Details:
- y = 1 – (sell_price / list_price)
- X: quantity, log(quantity+1), encoded product_id, country, customer_id
- `fit()`: learn stable encodings for categorical fields
- Handle missing list/sell price robustly (drop with warning)

Acceptance:
- Deterministic encodings; pandas-mode works; tests validate columns & shapes.
```

### Task C — Baseline model + MLflow logging

```
Goal: Implement `PricePointRegressor` using sklearn and MLflow logging.

Files:
- `src/pricing/models/price_point.py`
- `src/pricing/mlops/registry.py`
- `tests/unit/test_model_point.py`

Details:
- Use `RandomForestRegressor` with `.fit/.predict`
- Implement `registry.log_model()` to log params/metrics and save the model to MLflow; return model URI
- Respect `TRACKING_URI` env var; default to local MLflow

Acceptance:
- `pytest -q` passes locally without Spark.
```

### Task D — Make training script executable

```
Goal: Convert `notebooks/10_train_point.py` to a CLI entrypoint.

File:
- `notebooks/10_train_point.py`

Details:
- Args: `--config` (default `configs/bu-default.yaml`), `--holdout_days` (default 30)
- Steps: load config → read orders → FeatureBuilder → train model → compute RMSE/MAE on holdout → log to MLflow → register model
- Pandas-mode by default

Acceptance:
- Running `python notebooks/10_train_point.py --config configs/bu-default.yaml` succeeds locally.
```

## 5) Phase 2 — Scoring, aggregation, publish

### Task E — Line scoring

```
Goal: Load registered model and score line-level predictions.

Files:
- `src/pricing/inference/line_scorer.py`
- `notebooks/20_predict_lines.py`

Details:
- Implement `_load_model()` to load latest model for `bu_id` from MLflow registry
- `score(df)`: add `predicted_discount` and `recommended_sell_price = list_price * (1 - predicted_discount)`
- Notebook: read config; reuse orders as input for now; write CSV or Delta based on execution mode

Acceptance:
- Pandas-mode writes a CSV with expected columns.
```

### Task F — Deal aggregation (value-weighted) + rescale lines

```
Goal: Aggregate line-level recs to deal-level and rescale lines.

Files:
- `src/pricing/inference/deal_aggregator.py`
- `notebooks/30_aggregate_deal.py`
- `tests/unit/test_aggregate.py`

Details:
- Inputs: `deal_id, list_price, quantity, predicted_discount`
- Deal predicted discount = weighted average by `(list_price * quantity)`
- Return `df_deal` and `df_lines_adj` (scale line discounts to match deal target)

Acceptance:
- Unit test with two deals validates weighting and rescaling math.
```

### Task G — Post-process & publish

```
Goal: Apply guardrails and publish outputs.

File:
- `notebooks/40_postprocess_publish.py`

Details:
- Clip discounts to [0, 0.6]
- If floors provided, ensure `final_recommended_price >= floor`
- Save `final_` versions to configured outputs

Acceptance:
- Columns `clipped_discount` and `final_recommended_price` present.
```

## 6) Running locally

To run everything locally:

```bash
source .venv/bin/activate
pytest -q
black --check .

# Train
python notebooks/10_train_point.py --config configs/bu-default.yaml --holdout_days 30

# Score lines
python notebooks/20_predict_lines.py --config configs/bu-default.yaml

# Aggregate to deal
python notebooks/30_aggregate_deal.py --config configs/bu-default.yaml

# Post-process & publish
python notebooks/40_postprocess_publish.py --config configs/bu-default.yaml
```

## 7) Parallelization strategy

- Work per-task as described; each is mostly independent.
- Keep PRs small (≤ 300 lines) with tests.
- Branch naming: `feat/<task-key>-short-desc`, e.g. `feat/io-adapters`.

## 8) Tips for great Codex results

- Keep tasks small and file-scoped.
- Ask Codex to show a plan before applying diffs.
- Have Codex print shell commands rather than running them automatically.
- If a patch isn’t ideal, request a revision focusing on specific files.
- After applying a patch:

```bash
git add -A && git commit -m "feat: <summary>"
pytest -q
```

## 9) Troubleshooting

- **Spark import errors**: ensure pandas mode (`execution_mode: pandas`) and guard Spark imports.
- **MLflow URI**: set `MLFLOW_TRACKING_URI` env var if using a hosted server; local default works too.
- **CSV/Delta paths**: ensure YAML paths exist or point to a writable location.
- **Tests failing**: run `pytest -q -k <testname>` and ask Codex to fix just that test.

## 10) Next phases (stubs)

When Phase 1‑2 are green, consider:

- Implementing quantile or conformal ranges.
- Adding pseudo-deal context features.
- Integrating Databricks Asset Bundles for scheduling jobs.
- Building Chatbot and API interfaces for real-time recommendations.
- Adding drift detection and monitoring.

_End of guide_
