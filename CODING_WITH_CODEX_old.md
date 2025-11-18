# Price Desk – Development Order & Ownership Guide

This document describes **what to build first**, **who can work in parallel**, and **what “done” means** for each layer.

The initial focus is **Scenario A (orders-only)** for **one pilot BU (BU1)**:

* Only SAP orders history available.
* No reliable quote linkage, no win/loss labels.
* Single price model trained on all orders (mixture of negotiated + non-negotiated).
* Output: **line- and deal-level discount ranges** written to Delta for Power BI.

Later phases will extend to more BUs and more complex scenarios (quotes, gates, win-prob, mixture-of-experts).

---

## 0. Principles

* **Scenario-first:** start with the simplest real scenario (orders-only, BU1) and build a solid vertical slice.
* **Single codebase, multi-BU:** BU behavior controlled via **config**, not forks.
* **Spark+Pandas flexible:** feature & pipeline code must work on both, via `FrameAdapter`.
* **OOP & testable:** clear interfaces (`FeaturePipeline`, `Regressor`, etc.), unit tests for each piece.
* **MLOps-ready:** everything runs in Databricks jobs via **Asset Bundles** and is tracked in **MLflow**.

---

## 1. Phase 0 – Foundations

> Goal: scaffolding for logging, config, DataFrame abstraction, and MLflow usage.
> Dependencies: none.
> Parallelizable: yes.

### 0.1 Utilities: Logging & RunContext

**Files**

* `src/price_desk/utilities/commons.py`

**Tasks**

* Implement `get_logger(name: Optional[str])`:

  * Single `StreamHandler`, consistent log format.
  * `INFO` level default.
* Implement `RunContext` dataclass with `bu_id`, `env`, `run_id`, `extra`.

**Definition of Done**

* Importing `get_logger(__name__)` from different modules doesn’t produce duplicate log lines.
* Logs look like: `[2025-11-17 10:00:00] [INFO] [price_desk.some_module] message`.

---

### 0.2 DataFrame abstraction: `FrameAdapter`

**Files**

* `src/price_desk/utilities/frame_adapter.py`

**Tasks**

For `PandasFrameAdapter`:

* Implement:

  * `select(cols)`
  * `with_columns(new_cols: dict)`:

    * Support values as:

      * static value (broadcast),
      * function `(df: pd.DataFrame) -> pd.Series`.
  * `filter(expr)`:

    * bool series or callable `(df) -> bool series`.
  * `groupby_agg(keys, agg_spec)`:

    * `agg_spec` like `{"col": "sum", "other": "mean"}`.
  * `join(other, on, how, suffixes)`.
  * `to_pandas`, `from_pandas`.

For `SparkFrameAdapter`:

* Implement:

  * `select(cols)` → `df.select(*cols)`.
  * `with_columns(new_cols)`:

    * value can be `Column` or callable `(df) -> Column`.
  * `filter(expr)`:

    * string or `Column`.
  * `groupby_agg(keys, agg_spec)` using `groupBy().agg(...)`.
  * `join(other, on, how, suffixes)`.
  * `to_pandas`, `from_spark`.

**Definition of Done**

* `tests/test_frame_adapter.py`:

  * Build a small Pandas DF; perform a series of operations on both adapters:

    * select, new column, filter, groupby, join.
  * Compare Pandas outputs for Pandas vs Spark (after `.to_pandas()` and suitable sorting).
* Ready to be used by feature pipelines without knowing backend.

---

### 0.3 Config system

**Files**

* `src/price_desk/configs/schema.py`
* `src/price_desk/configs/loader.py`
* `src/price_desk/configs/bu_BU1.yaml` (first BU)

**Tasks**

* Implement Pydantic models:

  * `DataPaths`, `SegmentConfig`, `GuardrailConfig`, `TrainingParams`, `BUConfig`.
* Implement `load_bu_config(bu_id)` to load `bu_{bu_id}.yaml` into `BUConfig`.
* Create `bu_BU1.yaml` with:

  * BU id.
  * Orders table path.
  * Output tables for line & deal recs.
  * Segment keys (e.g., product_family, region, customer_tier, qty_band).
  * Basic guardrail settings.
  * Training params (target column, quantiles, experiment name).

**Definition of Done**

* `tests/test_config_schema.py` & `test_config_loader.py`:

  * Valid YAML loads without errors.
  * Missing required fields cause `ValidationError`.
* `load_bu_config("BU1")` works in a REPL / notebook.

---

### 0.4 MLflow helper

**Files**

* `src/price_desk/utilities/mlflow_utils.py`

**Tasks**

* Implement:

  * `mlflow_run(experiment_name, run_name=None)` context manager.
  * `log_params`, `log_metrics`, `log_artifact`.

**Definition of Done**

* `tests/test_mlflow_utils.py`:

  * With a local `file:` tracking URI, entering/leaving `mlflow_run` creates a run and records params/metrics.

---

## 2. Phase 1 – Scenario A vertical slice for BU1

> Goal: End-to-end pipeline for one BU using orders-only data, producing **discount ranges** into Delta.
> Dependencies: Phase 0 complete.
> Parallelizable: yes, across FE/Model/Policy/Aggregator/Jobs.

### 1.1 Feature Engineering – Orders-only

**Files**

* `src/price_desk/feature_engineering/base.py`
* `src/price_desk/feature_engineering/orders_features.py`

**Tasks**

* `FeaturePipeline` base:

  * `__init__(config: BUConfig)`
  * `fit(self, df: FrameAdapter) -> FeaturePipeline` (default no-op).
  * Abstract `transform(self, df: FrameAdapter) -> FrameAdapter`.
* `OrdersFeaturePipeline`:

  * Validate required columns:
    `order_id, order_line_id, customer_id, product_id, country, order_date, list_price, sell_price, quantity`.
  * Derive:

    * `discount_pct = 1 - sell_price / list_price` (with safe handling for zero list price).
    * `order_date` → optional `order_week`, `order_month` (for pseudo-deal context).
    * `qty_band` using simple bins (configurable thresholds).
    * `customer_tier` if there’s a rule (else leave TODO/placeholder column).
  * Pseudo deal context:

    * Example: by `(customer_id, order_week)` compute:

      * `cust_week_total_value` (sum list_price*qty),
      * `cust_week_line_count`.
    * Join back to line-level.
  * Ensure all **segment keys** in config exist (or raise error).

**Definition of Done**

* `tests/test_orders_features.py`:

  * Given a small Pandas DF:

    * `discount_pct` is correct (vectorized).
    * Derivative columns correct (qty bands, aggregates).
  * Same test passes using Spark via `SparkFrameAdapter`.

---

### 1.2 Model – GbmQuantileRegressor

**Files**

* `src/price_desk/modeling/base_models.py`
* `src/price_desk/modeling/gbm_regressor.py`

**Tasks**

* `Regressor` base:

  * Methods: `fit`, `predict`, `predict_quantiles`, `get_params`, `set_params`.
* `GbmQuantileRegressor`:

  * Constructor takes `base_params`, `quantiles`.
  * `fit(X, y)`:

    * Train point model.
    * For each quantile `q`, clone params with `objective="quantile"`, `alpha=q`, train and store.
  * `predict(X)` → point predictions.
  * `predict_quantiles(X, quantiles)`:

    * For each requested quantile, use appropriate model.
    * Return `DataFrame` with standardized column names (e.g., `disc_q_0_10`, `disc_q_0_50`, `disc_q_0_90`).

**Definition of Done**

* `tests/test_gbm_regressor.py`:

  * On toy data, `fit` + `predict` + `predict_quantiles` work, shapes correct.
  * For each row: `q_0_10 ≤ q_0_50 ≤ q_0_90`.
* Ready to be called by training/inference pipelines.

---

### 1.3 Training Pipeline – PriceTrainingPipeline

**Files**

* `src/price_desk/modeling/pipelines/price_training_pipeline.py`

**Tasks**

* Implement `PriceTrainingPipeline(config)`:

  * Accept a `FrameAdapter` (Spark/Pandas) containing raw orders.
  * Use `OrdersFeaturePipeline(config).fit(df).transform(df)` to compute features.
  * Convert to Pandas with `.to_pandas()`.
  * Split into train/test (use config’s `test_size`, `random_state`).
  * Instantiate `GbmQuantileRegressor` with config-provided params & quantiles.
  * Within `mlflow_run(experiment_name, run_name=f"price_train_{bu_id}")`:

    * Log model params.
    * Fit model.
    * Compute metrics (MAE/MAPE/RMSE on test).
    * Log metrics.
  * Return `PriceTrainingResult(model, feature_columns, target_column)`.

**Definition of Done**

* `tests/test_price_training_pipeline.py`:

  * Using a small PandasFrameAdapter with synthetic data, pipeline returns a result.
  * Model can predict on a holdout set.
* An MLflow run is created with params & metrics.

**Future**

* Add model registration here (to MLflow Model Registry under `price_rec_{bu_id}`).

---

### 1.4 Guardrail Policy – caps/floors on discount bands

**Files**

* `src/price_desk/model_inference/post_processing/policy.py`

**Tasks**

* Implement `GuardrailPolicy(config)`:

  * Input: Pandas DF with:

    * discount quantile columns (`disc_q_0_10`, `disc_q_0_50`, `disc_q_0_90`),
    * columns for **segment keys**.
  * Scenario A MVP:

    * Use config `cap_percentile` and `floor_percentile` as **static numeric caps** per BU (or temporarily global).
    * Clip all discount quantiles between [floor, cap].
    * Enforce monotonic quantiles after clipping.
  * Add flags:

    * `cap_hit` / `floor_hit` booleans or counters (optional now, useful later).

**Definition of Done**

* `tests/test_policy.py`:

  * Crafted input where some quantiles are outside bounds.
  * After `apply`, all quantiles within bounds and ordered `p10≤p50≤p90`.

**Future**

* Extend to data-driven caps/floors per segment based on historical actuals (Scenario B+).

---

### 1.5 Deal Aggregator – line → deal bands

**Files**

* `src/price_desk/model_inference/post_processing/aggregator.py`

**Tasks**

* Implement `DealAggregator(deal_id_col, value_cols=("list_price", "quantity"))`:

  * Expect columns:

    * `deal_id_col` (for now, `order_id`),
    * `list_price`, `quantity`,
    * `disc_q_*` columns.
  * For each line, compute `line_value = list_price * quantity`.
  * Group by `deal_id_col`, compute weighted average for each quantile:
    [
    D_{deal,q} = \frac{\sum_i disc_{i,q} \cdot value_i}{\sum_i value_i}
    ]
  * Return a DF with one row per deal, with `deal_disc_q_*`.

**Definition of Done**

* `tests/test_aggregator.py`:

  * Single-line deals: deal quantiles == line quantiles.
  * Two-line deal with different list_price*qty weights: check formula manually.

**Future**

* Add support for Monte-Carlo / copula-based aggregation (Phase 3).
* Add QP-based adjustment for consistent line-level prices matching chosen deal quantile.

---

### 1.6 Inference Pipeline – PriceInferencePipeline

**Files**

* `src/price_desk/model_inference/price_infer.py`

**Tasks**

* Implement `PriceInferencePipeline(config, model)`:

  * Input: `FrameAdapter` with raw orders (or a scoring subset).
  * Use `OrdersFeaturePipeline(config)` to `transform`.
  * Convert to Pandas.
  * Call `model.predict_quantiles` with configured quantiles.
  * Merge quantiles with features; add ids, segment keys.
  * Apply `GuardrailPolicy`.
  * Call `DealAggregator` to get deal-level bands.
  * Return `(line_df, deal_df)` as Pandas.

**Definition of Done**

* `tests/test_price_inference_pipeline.py`:

  * With a trained `GbmQuantileRegressor` on toy data, pipeline returns non-empty line & deal DFs.
  * Deal DF has correct IDs and discount columns.

---

### 1.7 Jobs – Databricks training & inference entrypoints

**Files**

* `src/jobs/training_pipeline.py`
* `src/jobs/inference_pipeline.py`

**Tasks**

**Training job**

* `main(bu_id)`:

  * Init `SparkSession`.
  * Load config.
  * Read orders table into Spark.
  * Wrap as `SparkFrameAdapter`.
  * Run `PriceTrainingPipeline.fit`.
  * (Optional now) register model in MLflow Model Registry with name `price_rec_{bu_id}`.

**Inference job**

* `main(bu_id, model_name)`:

  * Init `SparkSession`.
  * Load config.
  * Read scoring table (or orders table).
  * Wrap as `SparkFrameAdapter`.
  * Load model from MLflow (later via pyfunc or direct `GbmQuantileRegressor` artifact).
  * Run `PriceInferencePipeline`.
  * Convert Pandas outputs to Spark DFs.
  * Write to `data_paths.line_output_table` and `data_paths.deal_output_table`.

**Definition of Done**

* Both scripts runnable in a Databricks notebook cell (for dev) with test configs.
* On dev workspace, a small run completes and writes Delta outputs.

---

### 1.8 Asset Bundle resources (`resources/*.yml`)

**Files**

* `src/resources/model-workflow-resource.yml`
* `src/resources/batch-inference-workflow-resource.yml`

**Tasks**

* Training workflow:

  * Define job `price_training_BU1` (or parameterized job with `bu_id=BU1`).
  * Single task calling `python jobs/training_pipeline.py --bu_id=BU1`.
  * Monthly schedule.
* Inference workflow:

  * Define job `price_inference_BU1`.
  * Task calling `python jobs/inference_pipeline.py --bu_id=BU1 --model_name=price_rec_BU1`.
  * Daily or weekly schedule.

**Definition of Done**

* `databricks bundle validate` passes.
* `databricks bundle deploy --target=dev` creates jobs in workspace.

---

### 1.9 CI pipeline – basic

**Files**

* `.github/workflows/ci.yml`

**Tasks**

* Steps:

  * Set up Python.
  * `pip install -r src/requirements.txt`.
  * Lint (e.g., `ruff` or `flake8`).
  * Run `pytest src/tests`.
  * Optionally run `databricks bundle validate` (with env vars/secrets for CLI).

**Definition of Done**

* CI runs on PRs and main; fails if tests or lint fails.

---

## 3. Phase 2+ – Future work (after Scenario A vertical slice)

Once the above is working for BU1, you can:

1. **Add BU2, BU3, … (still Scenario A)**

   * Copy `bu_BU1.yaml` → `bu_BU2.yaml` etc.
   * Add jobs per BU in resource YAMLs.
   * Run pipelines on each BU’s orders table.

2. **Scenario C/D/B/E (quotes, gates, win-prob, mixtures)**

   * Introduce new modules sitting on top of same interfaces:

     * `negotiation_features.py`
     * `gate_classifier.py`
     * `sk_logistic.py` for win-prob
     * `mixture.py` for mixture-of-experts
     * extend `aggregator.py` for simulation/copula.
   * These are long-term and can be planned similarly.

---
