How we build this repo using the Codex – OpenAI’s coding agent VS Code extension

This guide shows how to delegate well-scoped tasks to Codex (OpenAI’s agent in VS Code) to implement our pricing-recs plan. It includes setup, guardrails, and ready-to-paste tasks you can run as “Codex: New Task”.

0) Prereqs

macOS with Python 3.10+

VS Code latest

ChatGPT Plus/Pro/Business (for Codex extension sign-in)

(Optional) GitHub Copilot for inline completions

Repo:

cd ~/work
git clone https://github.com/<your-org-or-user>/pricing-recs.git
cd pricing-recs
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt || pip install pytest pydantic mlflow black

1) Install & open Codex

In VS Code, install “Codex – OpenAI’s coding agent.”

Sign in with your OpenAI account.

Trust the workspace.

Open Command Palette → Codex: New Task.

You can also use the Codex side panel for conversational tasks.

2) House rules (paste into every task)

Copy/paste this block at the top of each Codex task so patches are safe and consistent:

Constraints:
- Modify only files in this repo and show a short plan first.
- Do not run shell/network commands without asking; print commands for me to run.
- Default to pandas mode; guard Spark imports (don’t break local runs).
- Add/modify unit tests; `pytest -q` must pass.
- Include type hints and docstrings.
- Keep changes minimal and scoped to listed files.

3) Project conventions

Python version: 3.10

Formatting: Black (line length 88)

Tests: pytest -q

Config: per-BU YAML in configs/

Pandas vs Spark: Both supported; pandas is default for local dev

MLflow: Log params/metrics; registry name pattern pricing_recs_<bu_id>

Outputs:

pandas → CSV paths in config

spark → Delta tables/paths in config

4) Phase 1 — Orders-only training pipeline
Task A — IO adapters + orders reader

Codex: New Task

Goal: Implement dual-mode DataFrame adapters and wire OrdersReader.

Files:
- Create src/pricing/io/adapters.py
- Edit src/pricing/io/connectors.py
- Create tests/unit/test_io_adapters.py

Details:
- Adapters: PdAdapter and SpAdapter implementing to_pandas(), to_spark(spark).
- OrdersReader.read(config): read CSV/Delta from config.input_table (infer by suffix; default CSV).
- If execution_mode == 'pandas' return PdAdapter; else SpAdapter.
- Guard Spark imports with a friendly error if Spark is unavailable.

Acceptance:
- pytest -q passes; round-trip test skips Spark if not installed.

Task B — FeatureBuilder v1 (orders-only)
Goal: Implement FeatureBuilder.fit/transform for orders-only.

Files:
- src/pricing/features/feature_builder.py
- tests/unit/test_features.py

Details:
- y = 1 - (sell_price / list_price)
- X: quantity, log(quantity+1), product_id_enc, country_enc, customer_id_enc
- fit(): learn stable encodings for categorical fields
- Handle missing list/sell price robustly (drop with warning)

Acceptance:
- Deterministic encodings; pandas-mode works; tests validate columns & shapes.

Task C — Baseline model + MLflow logging
Goal: Implement PricePointRegressor (sklearn) and MLflow registry hook.

Files:
- src/pricing/models/price_point.py
- src/pricing/mlops/registry.py
- tests/unit/test_model_point.py

Details:
- Use RandomForestRegressor; implement fit/predict
- registry.log_model(): start run if needed, log params/metrics, mlflow.sklearn.log_model; return model URI
- Respect TRACKING_URI env var if present; otherwise local default

Acceptance:
- pytest -q passes locally with no Spark dependency.

Task D — Make training script executable
Goal: Convert notebooks/10_train_point.py to a CLI entrypoint.

File:
- notebooks/10_train_point.py

Details:
- Args: --config (default configs/bu-default.yaml), --holdout_days (default 30)
- Steps: load config → read orders → FeatureBuilder → train model → compute RMSE/MAE on holdout → log to MLflow → register model as pricing_recs_<bu_id>
- pandas-mode by default

Acceptance:
- Running: `python notebooks/10_train_point.py --config configs/bu-default.yaml` succeeds locally.

5) Phase 2 — Scoring, aggregation, publish
Task E — Line scoring
Goal: Load registered model and score line-level predictions.

Files:
- src/pricing/inference/line_scorer.py
- notebooks/20_predict_lines.py

Details:
- LineScorer._load_model(): load latest for bu_id from MLflow Registry (or MODEL_URI env fallback)
- score(df): add predicted_discount and recommended_sell_price = list_price*(1 - predicted_discount)
- 20_predict_lines.py: read config; reuse orders as input for now; write:
  * pandas: CSV to config.output_table_line
  * spark: Delta to config.output_table_line

Acceptance:
- Pandas-mode writes a CSV with expected columns.

Task F — Deal aggregation (value-weighted) + rescale lines
Goal: Aggregate to deal-level and ensure consistency.

Files:
- src/pricing/inference/deal_aggregator.py
- notebooks/30_aggregate_deal.py
- tests/unit/test_aggregate.py

Details:
- Inputs: [deal_id, list_price, quantity, predicted_discount]
- Deal predicted discount = weighted avg by (list_price * quantity)
- Return df_deal and df_lines_adj (scale line discounts to match the deal target)

Acceptance:
- Unit test with 2 deals validates weighting and rescaling math.

Task G — Post-process & publish
Goal: Guardrails and final outputs.

File:
- notebooks/40_postprocess_publish.py

Details:
- Clip discounts to [0, 0.6]
- If floors provided, ensure final_recommended_price >= floor
- Save "final_" versions to configured outputs (CSV/Delta)

Acceptance:
- Columns: clipped_discount, final_recommended_price present.

6) Running locally
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

7) Parallelization strategy

Work per-task above; each is ~independent.

Keep PRs small (≤ ~300 lines) with tests.

Branch naming: feat/<task-key>-short-desc
Example: feat/io-adapters, feat/featurebuilder-v1.

8) Tips for great Codex results

Keep tasks small and file-scoped.

Ask Codex to show a plan before applying diffs.

If it wants to run commands, have it print them; you run them in the terminal.

If a patch isn’t ideal, say “revise only X and Y; leave Z unchanged.”

After applying a patch:

git add -A && git commit -m "feat: <summary>"
pytest -q

9) Troubleshooting

Spark import errors: ensure pandas mode (config execution_mode: pandas) and that imports are guarded.

MLflow URI: set export MLFLOW_TRACKING_URI=<your-uri> if using a hosted server. Local default works too.

CSV/Delta paths: ensure your configs/*.yaml paths exist or point to a writable location.

Tests failing: run pytest -q -k <testname> and ask Codex to fix just that test.

10) Next phases (stubs)

When Phase 1–2 are green, add:

Quantile or conformal ranges for line + deal (QRF or conformalized RF).

Pseudo-deal context features (rolling stats by inferred quote window).

Databricks Asset Bundles job definitions per BU (dev/stage/prod).

Use the same Codex pattern: one small task per file + tests.
