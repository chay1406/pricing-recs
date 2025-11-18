---
name: "Scenario A – BU Vertical Slice (Orders-only)"
about: "Implement end-to-end Scenario A (orders-only) pricing pipeline for a BU"
title: "[Scenario A] BU1 vertical slice – orders-only price bands"
labels: ["scenario-a", "bu1", "ml", "databricks", "infra"]
assignees: []
---

## Overview

Goal: Deliver an **end-to-end pipeline** for **Scenario A (orders-only)** for one BU (start with **BU1**):

- Use **SAP orders** only (no quotes/win-loss).
- Train a **single regression model** to predict **line-level discount ranges** (P10/P50/P90).
- Apply **guardrails** and aggregate to **deal-level ranges**.
- Write **line & deal recommendations** to **Delta tables** for use in **Power BI**.
- Use **Databricks Asset Bundles** for jobs and **GitHub Actions** for CI.

> Replace `BU1` with the target BU where needed.

---

## Phase 0 – Foundations (shared)

### 0.1 Logging & Run Context

- [ ] Implement `get_logger` in `src/price_desk/utilities/commons.py`
  - [ ] Configure a single `StreamHandler` with a consistent format: `[time] [level] [name] message`
  - [ ] Ensure multiple calls don’t create duplicate handlers
- [ ] Implement `RunContext` dataclass (`bu_id`, `env`, `run_id`, `extra`)
- [ ] Add basic unit test(s) in `src/tests/test_commons.py`
  - [ ] Verify log format
  - [ ] Verify no duplicate handlers

### 0.2 DataFrame Abstraction – `FrameAdapter`

**Files:** `src/price_desk/utilities/frame_adapter.py`

- [ ] Implement `PandasFrameAdapter`
  - [ ] `select(cols)`
  - [ ] `with_columns(new_cols)` with:
    - [ ] static values
    - [ ] callables `(df: pd.DataFrame) -> pd.Series`
  - [ ] `filter(expr)` with:
    - [ ] boolean Series
    - [ ] callable `(df) -> bool Series`
  - [ ] `groupby_agg(keys, agg_spec)` using `groupby().agg()`
  - [ ] `join(other, on, how, suffixes)` using `merge`
  - [ ] `to_pandas`, `from_pandas`
- [ ] Implement `SparkFrameAdapter`
  - [ ] `select(cols)` → `df.select(*cols)`
  - [ ] `with_columns(new_cols)` with:
    - [ ] Spark `Column`
    - [ ] callable `(df) -> Column`
  - [ ] `filter(expr)` with:
    - [ ] string SQL expr
    - [ ] `Column`
  - [ ] `groupby_agg(keys, agg_spec)` using `groupBy().agg(...)`
  - [ ] `join(other, on, how, suffixes)`
  - [ ] `to_pandas` → `df.toPandas()`
  - [ ] `from_spark`
- [ ] Add tests in `src/tests/test_frame_adapter.py`
  - [ ] Create small Pandas DF fixture
  - [ ] Apply sequence of operations on PandasFrameAdapter
  - [ ] Apply same operations on equivalent Spark DF via SparkFrameAdapter
  - [ ] Compare resulting Pandas outputs (after sorting & column ordering)

### 0.3 Config System

**Files:**  
`src/price_desk/configs/schema.py`  
`src/price_desk/configs/loader.py`  
`src/price_desk/configs/bu_BU1.yaml`

- [ ] Define Pydantic models:
  - [ ] `DataPaths` (orders_table, scoring_table?, line_output_table, deal_output_table, diagnostics_table?)
  - [ ] `SegmentConfig` (segment keys)
  - [ ] `GuardrailConfig` (cap_percentile, floor_percentile, etc.)
  - [ ] `TrainingParams` (target_column, test_size, random_state, quantiles, experiment_name)
  - [ ] `BUConfig` (bu_id, data_paths, segments, guardrails, training, extra)
- [ ] Implement `load_bu_config(bu_id)` in `loader.py`
  - [ ] Build path like `bu_{bu_id}.yaml`
  - [ ] Load with `yaml.safe_load`
  - [ ] Construct `BUConfig`
- [ ] Create `bu_BU1.yaml` with:
  - [ ] `bu_id: "BU1"`
  - [ ] Orders table path
  - [ ] Line & deal output table names
  - [ ] Segment keys (e.g., `product_family`, `region`, `customer_tier`, `qty_band`)
  - [ ] Guardrail settings (cap/floor percentiles)
  - [ ] Training params (target = `discount_pct`, quantiles `[0.1, 0.5, 0.9]`, experiment name)
- [ ] Add tests:
  - [ ] `src/tests/test_config_schema.py` – valid config loads, invalid config fails
  - [ ] `src/tests/test_config_loader.py` – `load_bu_config("BU1")` returns a valid BUConfig

### 0.4 MLflow Helpers

**Files:** `src/price_desk/utilities/mlflow_utils.py`

- [ ] Implement `mlflow_run(experiment_name, run_name=None)` context manager
- [ ] Implement `log_params`, `log_metrics`, `log_artifact`
- [ ] Add tests in `src/tests/test_mlflow_utils.py` using a local `file:` tracking URI
  - [ ] Verify a run is created
  - [ ] Verify params and metrics logged

---

## Phase 1 – Scenario A Vertical Slice (BU1)

### 1.1 Feature Engineering – Orders-only

**Files:**  
`src/price_desk/feature_engineering/base.py`  
`src/price_desk/feature_engineering/orders_features.py`

- [ ] Implement `FeaturePipeline` base:
  - [ ] `__init__(config: BUConfig)`
  - [ ] Default `fit(self, df: FrameAdapter) -> FeaturePipeline` (no-op)
  - [ ] Abstract `transform(self, df: FrameAdapter) -> FrameAdapter`
- [ ] Implement `OrdersFeaturePipeline`:
  - [ ] Validate required columns:
    - [ ] `order_id`, `order_line_id`, `customer_id`, `product_id`, `country`, `order_date`, `list_price`, `sell_price`, `quantity`
  - [ ] Derive core fields:
    - [ ] `discount_pct = 1 - sell_price / list_price` (handle list_price = 0 safely)
    - [ ] `order_week` / `order_month` (optional)
    - [ ] `qty_band` via simple thresholds (config-driven or hard-coded for now)
    - [ ] `customer_tier` (rule-based if available; else placeholder)
  - [ ] Derive pseudo deal context:
    - [ ] For `(customer_id, order_week)` compute:
      - [ ] `cust_week_total_value = sum(list_price * quantity)`
      - [ ] `cust_week_line_count`
    - [ ] Join aggregates back to line-level
  - [ ] Ensure all **segment keys** in BU config exist as columns; raise if missing
- [ ] Add tests in `src/tests/test_orders_features.py`:
  - [ ] Use `PandasFrameAdapter` with small fixture; validate:
    - [ ] `discount_pct`
    - [ ] `qty_band`
    - [ ] aggregates correctness
  - [ ] Same test with Spark via `SparkFrameAdapter`

### 1.2 Modeling – `GbmQuantileRegressor`

**Files:**  
`src/price_desk/modeling/base_models.py`  
`src/price_desk/modeling/gbm_regressor.py`

- [ ] Implement `Regressor` base with methods:
  - [ ] `fit(X, y)`
  - [ ] `predict(X)`
  - [ ] `predict_quantiles(X, quantiles)`
  - [ ] `get_params()`
  - [ ] `set_params(**params)`
- [ ] Implement `GbmQuantileRegressor`:
  - [ ] Accept `base_params` and `quantiles`
  - [ ] In `fit`:
    - [ ] Train point model (typical regression objective)
    - [ ] For each quantile `q`, clone params with `objective="quantile"`, `alpha=q`, train and store
  - [ ] In `predict`:
    - [ ] Use point model
  - [ ] In `predict_quantiles`:
    - [ ] Use corresponding quantile models
    - [ ] Return `DataFrame` with standardized column names (e.g., `disc_q_0_10`, `disc_q_0_50`, `disc_q_0_90`)
- [ ] Add tests in `src/tests/test_gbm_regressor.py`:
  - [ ] On toy data, `fit` runs without error
  - [ ] `predict` and `predict_quantiles` produce correct shapes
  - [ ] For each row: `q_0_10 ≤ q_0_50 ≤ q_0_90`

### 1.3 Training Pipeline – `PriceTrainingPipeline`

**File:** `src/price_desk/modeling/pipelines/price_training_pipeline.py`

- [ ] Implement `PriceTrainingPipeline(config: BUConfig)`:
  - [ ] Accept `FrameAdapter` with raw orders
  - [ ] Use `OrdersFeaturePipeline(config).fit(df).transform(df)`
  - [ ] Convert to Pandas with `.to_pandas()`
  - [ ] Split into train/test using config’s `test_size`, `random_state`
  - [ ] Instantiate `GbmQuantileRegressor` from config
  - [ ] Within `mlflow_run(config.training.experiment_name, run_name=f"price_train_{config.bu_id}")`:
    - [ ] Log model params
    - [ ] Fit model
    - [ ] Compute metrics (MAE, MAPE, RMSE on test)
    - [ ] Log metrics
  - [ ] Return `PriceTrainingResult(model, feature_columns, target_column)`
- [ ] Add tests in `src/tests/test_price_training_pipeline.py`:
  - [ ] With small PandasFrameAdapter, pipeline returns `PriceTrainingResult`
  - [ ] Model can predict on simple holdout

> Optional (later in same issue or a follow-up):
> - [ ] Register trained model in MLflow Model Registry (`price_rec_BU1`)

### 1.4 Guardrail Policy – Caps/Floors

**File:** `src/price_desk/model_inference/post_processing/policy.py`

- [ ] Implement `GuardrailPolicy(config: BUConfig)`:
  - [ ] Expects DF with:
    - [ ] discount quantiles: `disc_q_0_10`, `disc_q_0_50`, `disc_q_0_90`
    - [ ] segment key columns from config
  - [ ] Scenario A MVP:
    - [ ] Use static cap and floor from `GuardrailConfig` (per BU)
    - [ ] Clip all discount quantiles to [floor, cap]
    - [ ] Enforce monotonic quantiles after clipping (ensure p10 ≤ p50 ≤ p90)
  - [ ] Optionally add `cap_hit` / `floor_hit` flags or counts
- [ ] Add tests in `src/tests/test_policy.py`:
  - [ ] Construct DF with values outside bounds
  - [ ] After `apply`, quantiles within bounds and monotonic

### 1.5 Deal Aggregator – Line → Deal Ranges

**File:** `src/price_desk/model_inference/post_processing/aggregator.py`

- [ ] Implement `DealAggregator(deal_id_col: str, value_cols=("list_price", "quantity"))`:
  - [ ] Expects:
    - [ ] `deal_id_col` (e.g., `order_id`),
    - [ ] `list_price`, `quantity`,
    - [ ] `disc_q_*` columns.
  - [ ] Compute `line_value = list_price * quantity`
  - [ ] Group by deal id and compute **value-weighted average** per quantile:
    - [ ] `deal_disc_q_*`
- [ ] Add tests in `src/tests/test_aggregator.py`:
  - [ ] Single-line deals: deal quantiles == line quantiles
  - [ ] Multi-line deals: verify weighted average manually

### 1.6 Inference Pipeline – `PriceInferencePipeline`

**File:** `src/price_desk/model_inference/price_infer.py`

- [ ] Implement `PriceInferencePipeline(config, model)`:
  - [ ] Accept `FrameAdapter` with raw scoring data (orders or scoring subset)
  - [ ] Use `OrdersFeaturePipeline(config)` to transform
  - [ ] Convert to Pandas
  - [ ] Call `model.predict_quantiles` with BU-configured quantiles
  - [ ] Merge quantiles with features and ids
  - [ ] Apply `GuardrailPolicy`
  - [ ] Use `DealAggregator` to compute deal-level bands
  - [ ] Return `(line_df, deal_df)` as Pandas
- [ ] Add tests in `src/tests/test_price_inference_pipeline.py`:
  - [ ] With a trained toy model, pipeline returns line & deal DFs
  - [ ] Deal DFs have expected discount columns and IDs

### 1.7 Databricks Jobs – Training & Inference Entry Points

**Files:**  
`src/jobs/training_pipeline.py`  
`src/jobs/inference_pipeline.py`

**Training job**

- [ ] Implement `main(bu_id)`:
  - [ ] Initialize `SparkSession`
  - [ ] Load BU config
  - [ ] Read orders table into Spark DF
  - [ ] Wrap with `SparkFrameAdapter`
  - [ ] Run `PriceTrainingPipeline.fit`
  - [ ] (Optional) register model in MLflow registry as `price_rec_{bu_id}`
- [ ] Add minimal tests (mocked where possible) or manual notebook validation

**Inference job**

- [ ] Implement `main(bu_id, model_name)`:
  - [ ] Initialize `SparkSession`
  - [ ] Load BU config
  - [ ] Read scoring table or orders table
  - [ ] Wrap with `SparkFrameAdapter`
  - [ ] Load model from MLflow registry / pyfunc (to be wired later)
  - [ ] Run `PriceInferencePipeline`
  - [ ] Convert Pandas outputs back to Spark DFs
  - [ ] Write to configured line & deal output tables (Delta)
- [ ] Manual test in Databricks notebook (dev workspace)
  - [ ] Confirm output tables exist and have expected schema

### 1.8 Asset Bundles – Jobs Definitions

**Files:**  
`src/resources/model-workflow-resource.yml`  
`src/resources/batch-inference-workflow-resource.yml`  
`src/databricks.yml`

- [ ] Define training job (`price_training_BU1`):
  - [ ] Task: `python jobs/training_pipeline.py --bu_id=BU1`
  - [ ] Reasonable cluster config (single-node if fine)
  - [ ] Monthly schedule
- [ ] Define inference job (`price_inference_BU1`):
  - [ ] Task: `python jobs/inference_pipeline.py --bu_id=BU1 --model_name=price_rec_BU1`
  - [ ] Daily/weekly schedule
- [ ] Add jobs to `databricks.yml` under appropriate target(s)
- [ ] Validate bundle:
  - [ ] `databricks bundle validate`
  - [ ] `databricks bundle deploy --target=dev` (smoke test)

### 1.9 CI Pipeline – Basic

**File:** `.github/workflows/ci.yml`

- [ ] Set up:
  - [ ] Python environment
  - [ ] `pip install -r src/requirements.txt`
  - [ ] Lint step (e.g., `ruff` or `flake8`)
  - [ ] Tests: `pytest src/tests`
  - [ ] (Optional) `databricks bundle validate` (with secrets)
- [ ] Confirm CI passes on PRs and `main`

---

## Acceptance Criteria for This Issue

- [ ] Scenario A vertical slice implemented for **BU1**:
  - [ ] Training job runs successfully, logs to MLflow.
  - [ ] Inference job runs successfully, writes **line** & **deal** Delta tables.
- [ ] Feature engineering, modeling, policy, and aggregation are properly unit-tested.
- [ ] CI pipeline is green (lint + tests).
- [ ] Databricks bundle validates and deploys jobs to the dev workspace.

---

> When this issue is complete, we should be able to:
> - point Power BI at `price_recs_line` & `price_recs_deal` for BU1,
> - show historical recommendation ranges for a given period,
> - and iterate on guardrails & features with business stakeholders.
