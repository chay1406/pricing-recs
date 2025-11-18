## EPIC 1 – Foundational Utilities & Project Scaffolding

> Goal: Common utilities (logging, DataFrame abstraction, MLflow helpers) so all other work can build on a consistent base.

---

### Story 1.1 – Implement common logging & run context

**As** a developer
**I want** a shared logging setup and run context
**So that** all modules log consistently and runs are traceable.

**Sub-tasks**

* **Sub-task 1.1.1 – Implement `commons` module**

  * Create `src/price_desk/utilities/commons.py`.
  * Implement `get_logger(name: Optional[str])`:

    * Use Python `logging` with a single `StreamHandler`.
    * Set a consistent format: `[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s`.
    * Ensure multiple calls do **not** attach duplicate handlers.
  * Implement `RunContext` dataclass:

    * Fields: `bu_id: str`, `env: str = "dev"`, `run_id: Optional[str]`, `extra: Dict[str, Any]`.

* **Sub-task 1.1.2 – Add unit tests for logging**

  * Create `src/tests/test_commons.py`.
  * Test that:

    * `get_logger()` returns a logger with one handler.
    * Calling `get_logger()` multiple times does not increase handler count.
    * Log message format contains time, level, and logger name.

---

### Story 1.2 – Implement Spark/Pandas `FrameAdapter`

**As** a data engineer / ML engineer
**I want** a unified DataFrame abstraction
**So that** feature and pipeline code can run on both Pandas and Spark without branching logic.

**Sub-tasks**

* **Sub-task 1.2.1 – Implement `FrameAdapter` base**

  * Define `FrameAdapter` abstract class in `src/price_desk/utilities/frame_adapter.py` with:

    * `select(self, cols)`
    * `with_columns(self, new_cols: Dict[str, Any])`
    * `filter(self, expr)`
    * `groupby_agg(self, keys: List[str], agg_spec: Dict[str, Any])`
    * `join(...)`
    * `to_pandas(self)`
    * `@classmethod from_pandas`, `@classmethod from_spark`.

* **Sub-task 1.2.2 – Implement `PandasFrameAdapter`**

  * Wrap `pd.DataFrame` in `PandasFrameAdapter`.
  * Implement:

    * `select(cols)` → basic column subset.
    * `with_columns(new_cols)`:

      * Accept static values or callables `(df: pd.DataFrame) -> pd.Series`.
    * `filter(expr)`:

      * Support boolean Series or callable `(df) -> bool Series`.
    * `groupby_agg(keys, agg_spec)`:

      * Use `groupby(keys).agg(agg_spec).reset_index()`.
    * `join(other, on, how, suffixes)` → `df.merge`.
    * `to_pandas` returns underlying DF; `from_pandas` wraps new DF.

* **Sub-task 1.2.3 – Implement `SparkFrameAdapter`**

  * Wrap `pyspark.sql.DataFrame` in `SparkFrameAdapter`.
  * Implement:

    * `select(cols)` → `df.select(*cols)`.
    * `with_columns(new_cols)`:

      * Value is a `Column` or callable `(df) -> Column`.
      * Add columns via `withColumn`.
    * `filter(expr)`:

      * If string, pass to `df.filter(expr)`.
      * If `Column`, also use `df.filter(expr)`.
    * `groupby_agg(keys, agg_spec)`:

      * Use `df.groupBy(keys).agg(...)` with appropriate Spark functions.
    * `join(other, on, how, suffixes)` → `df.join(...)`.
    * `to_pandas()` → `df.toPandas()`.
    * `from_spark(df)` wraps the Spark DF.

* **Sub-task 1.2.4 – Add adapter tests**

  * Create `src/tests/test_frame_adapter.py`.
  * Build a small Pandas fixture DF.
  * For Pandas:

    * Wrap in `PandasFrameAdapter` and apply sequence:

      * `select`, `with_columns`, `filter`, `groupby_agg`, `join`.
  * For Spark:

    * Create equivalent Spark DF.
    * Wrap in `SparkFrameAdapter` and apply the same sequence.
  * Compare Pandas outputs (after `.to_pandas()`, sorted, columns aligned).
  * Confirm outputs match.

---

### Story 1.3 – Implement MLflow helper utilities

**As** an ML engineer
**I want** a consistent way to start MLflow runs and log params/metrics
**So that** all experiments are tracked in a uniform way.

**Sub-tasks**

* **Sub-task 1.3.1 – Implement `mlflow_utils`**

  * Create `src/price_desk/utilities/mlflow_utils.py`.
  * Implement:

    * `mlflow_run(experiment_name, run_name=None)` context manager:

      * Sets experiment, starts run, logs run start/end with `get_logger`.
    * `log_params(params: Dict[str, Any])`.
    * `log_metrics(metrics: Dict[str, float], step: Optional[int] = None)`.
    * `log_artifact(path, artifact_path=None)`.

* **Sub-task 1.3.2 – Add MLflow helper tests**

  * Create `src/tests/test_mlflow_utils.py`.
  * Set a local `file:` tracking URI in test.
  * Within `mlflow_run`, log dummy params and metrics.
  * Assert that:

    * A run was created.
    * The run contains logged params and metrics.

---

## EPIC 2 – Configuration & BU Onboarding

> Goal: Config system to define BU-specific paths, segment keys, guardrails, and training settings.

---

### Story 2.1 – Define Pydantic config schema

**As** a developer
**I want** a typed config model
**So that** misconfigurations for BUs are caught early.

**Sub-tasks**

* **Sub-task 2.1.1 – Implement schema models**

  * In `src/price_desk/configs/schema.py`, define:

    * `DataPaths` with:

      * `orders_table`, `scoring_table` (optional), `line_output_table`, `deal_output_table`, `diagnostics_table` (optional).
    * `SegmentConfig` with:

      * `keys: List[str]`.
    * `GuardrailConfig` with:

      * `cap_percentile: float`,
      * `floor_percentile: float`,
      * `widen_factor_negotiated: float` (future use),
      * `widen_factor_default: float`.
    * `TrainingParams` with:

      * `target_column`, `test_size`, `random_state`, `quantiles`, `experiment_name`.
    * `BUConfig` with:

      * `bu_id`, `data_paths`, `segments`, `guardrails`, `training`, `extra: Dict[str, Any]`.

* **Sub-task 2.1.2 – Add schema tests**

  * Create `src/tests/test_config_schema.py`.
  * Create a minimal valid config dict, instantiate `BUConfig`, assert no error.
  * Create an invalid config (missing required fields), assert `ValidationError`.

---

### Story 2.2 – Implement BU config loader and BU1 config

**As** a data platform engineer
**I want** to load BU-specific YAML configs
**So that** the same codebase can support multiple BUs.

**Sub-tasks**

* **Sub-task 2.2.1 – Implement config loader**

  * In `src/price_desk/configs/loader.py`:

    * Implement `load_bu_config(bu_id: str, config_dir: Optional[Path] = None) -> BUConfig`.
    * Build path `config_dir / f"bu_{bu_id}.yaml"`.
    * Load YAML with `yaml.safe_load`.
    * Instantiate and return `BUConfig`.

* **Sub-task 2.2.2 – Create BU1 config file**

  * `src/price_desk/configs/bu_BU1.yaml`:

    * `bu_id: "BU1"`.
    * `data_paths`:

      * `orders_table: "<workspace-path>.orders_BU1"` (to be replaced with real table).
      * `line_output_table`, `deal_output_table` names.
    * `segments.keys` (e.g., `["product_family", "region", "customer_tier", "qty_band"]`).
    * `guardrails`:

      * `cap_percentile: 0.75`,
      * `floor_percentile: 0.10`.
    * `training`:

      * `target_column: "discount_pct"`,
      * `test_size: 0.2`,
      * `random_state: 42`,
      * `quantiles: [0.1, 0.5, 0.9]`,
      * `experiment_name: "price_rec_BU1"`.

* **Sub-task 2.2.3 – Add loader tests**

  * Create `src/tests/test_config_loader.py`.
  * Ensure `load_bu_config("BU1")` returns a valid `BUConfig`.
  * Assert that `bu_id`, paths, and quantiles are correctly loaded.

---

## EPIC 3 – Feature Engineering (Orders-only, Scenario A)

> Goal: Turn raw SAP orders into model-ready features using a consistent pipeline.

---

### Story 3.1 – Implement generic `FeaturePipeline` base class

**As** an ML engineer
**I want** a standard interface for feature pipelines
**So that** different scenarios/BUs can plug in their own implementations.

**Sub-tasks**

* **Sub-task 3.1.1 – Implement `FeaturePipeline` base**

  * In `src/price_desk/feature_engineering/base.py`:

    * `__init__(self, config: BUConfig)` stores config.
    * Default `fit(self, df: FrameAdapter) -> FeaturePipeline` returns `self`.
    * Abstract `transform(self, df: FrameAdapter) -> FrameAdapter`.

* **Sub-task 3.1.2 – Add base class tests**

  * `src/tests/test_feature_base.py`:

    * Create a dummy subclass implementing `transform`.
    * Ensure `FeaturePipeline` cannot be instantiated directly (abstract).

---

### Story 3.2 – Implement `OrdersFeaturePipeline` for Scenario A

**As** a data scientist
**I want** to compute standard pricing features from orders-only data
**So that** the model sees consistent inputs across BUs.

**Sub-tasks**

* **Sub-task 3.2.1 – Implement core feature transformations**

  * In `src/price_desk/feature_engineering/orders_features.py`:

    * Validate presence of required columns:

      * `order_id`, `order_line_id`, `customer_id`, `product_id`, `country`, `order_date`, `list_price`, `sell_price`, `quantity`.
    * Compute:

      * `discount_pct = 1 - sell_price / list_price` (with safe handling when `list_price == 0`).
      * `order_week` and/or `order_month` from `order_date`.
      * `qty_band` based on simple thresholds (configurable or initial hard-coded).
      * Optional `customer_tier` from rules or leave for future if input exists.
    * Pseudo deal context:

      * Group by `(customer_id, order_week)` to compute:

        * `cust_week_total_value = sum(list_price * quantity)`.
        * `cust_week_line_count = count(order_line_id)`.
      * Join these aggregates back to line level.

* **Sub-task 3.2.2 – Ensure segment keys exist**

  * Use `config.segments.keys` to verify each column exists after `transform`.
  * If a required segment key is missing, raise a clear error.

* **Sub-task 3.2.3 – Add feature pipeline tests**

  * `src/tests/test_orders_features.py`:

    * Using `PandasFrameAdapter`:

      * Create small synthetic DF.
      * Run `OrdersFeaturePipeline(config).transform`.
      * Check:

        * `discount_pct` values are correct.
        * `qty_band` labels match expectation.
        * Aggregate columns (`cust_week_total_value`, etc.) correct.
    * Repeat same test with Spark via `SparkFrameAdapter`.

---

## EPIC 4 – Modeling & Training Pipeline (Scenario A)

> Goal: Train a BU-specific price regression model with quantile outputs and log experiments in MLflow.

---

### Story 4.1 – Implement abstract `Regressor` interface

**As** an ML engineer
**I want** a generic regression interface
**So that** training/inference pipelines are decoupled from a specific algorithm.

**Sub-tasks**

* **Sub-task 4.1.1 – Implement `Regressor` base**

  * In `src/price_desk/modeling/base_models.py`, define abstract methods:

    * `fit(X: pd.DataFrame, y) -> Regressor`
    * `predict(X: pd.DataFrame) -> pd.Series`
    * `predict_quantiles(X: pd.DataFrame, quantiles: Iterable[float]) -> pd.DataFrame`
    * `get_params() -> Dict[str, Any>`
    * `set_params(**params) -> Regressor`
  * Add basic documentation on usage.

* **Sub-task 4.1.2 – Add interface tests**

  * `src/tests/test_base_models.py`:

    * Define a dummy subclass that implements methods in a trivial way.
    * Ensure instantiation and calls behave as expected.

---

### Story 4.2 – Implement `GbmQuantileRegressor` for discount ranges

**As** an ML engineer
**I want** a LightGBM-based regressor with quantile predictions
**So that** I can produce discount ranges (P10/P50/P90) per line.

**Sub-tasks**

* **Sub-task 4.2.1 – Implement LightGBM-based `GbmQuantileRegressor`**

  * In `src/price_desk/modeling/gbm_regressor.py`:

    * Constructor accepts `base_params: Dict[str, Any]`, `quantiles: Iterable[float]`.
    * Maintain:

      * `self._point_model: LGBMRegressor`.
      * `self._quantile_models: Dict[float, LGBMRegressor]`.
    * `fit(X, y)`:

      * Train `self._point_model` with standard regression objective.
      * For each quantile `q`:

        * Clone base params.
        * Set `objective="quantile"`, `alpha=q`.
        * Train and store in `_quantile_models[q]`.
    * `predict(X)`:

      * Return point predictions as a `pd.Series`.
    * `predict_quantiles(X, quantiles)`:

      * For each requested quantile:

        * Use the corresponding model.
        * Collect into a `pd.DataFrame` with well-defined column names (`disc_q_0_10`, etc.).
    * `get_params` / `set_params`:

      * Manage `base_params` and `quantiles`.

* **Sub-task 4.2.2 – Add `GbmQuantileRegressor` tests**

  * `src/tests/test_gbm_regressor.py`:

    * Build a small synthetic dataset.
    * Fit model and check:

      * `predict` and `predict_quantiles` run without error.
      * Output shapes are expected.
      * For each row, `q10 ≤ q50 ≤ q90`.

---

### Story 4.3 – Implement `PriceTrainingPipeline` for BU1

**As** a data scientist
**I want** a training pipeline that goes orders → features → model → MLflow
**So that** I can retrain models per BU in a standardized way.

**Sub-tasks**

* **Sub-task 4.3.1 – Implement pipeline logic**

  * In `src/price_desk/modeling/pipelines/price_training_pipeline.py`:

    * Class `PriceTrainingPipeline(config: BUConfig)`:

      * Method `fit(self, raw_df: FrameAdapter) -> PriceTrainingResult`:

        * Instantiate `OrdersFeaturePipeline(config)` and `fit` (if needed).
        * `transform` raw data to feature DF.
        * Convert to Pandas via adapter `.to_pandas()`.
        * Separate `X` and `y` using `config.training.target_column`.
        * Train/test split using `test_size` and `random_state`.
        * Instantiate `GbmQuantileRegressor` with config-defined quantiles and params.
        * Wrap training in `mlflow_run(config.training.experiment_name, run_name=f"price_train_{config.bu_id}")`:

          * Log model parameters.
          * Fit the model.
          * Compute metrics (MAE/MAPE/RMSE) on test set.
          * Log metrics.
        * Return `PriceTrainingResult(model, feature_columns, target_column)`.

* **Sub-task 4.3.2 – Add training pipeline tests**

  * `src/tests/test_price_training_pipeline.py`:

    * Use `PandasFrameAdapter` with synthetic data and minimal BU config.
    * Run `PriceTrainingPipeline.fit`.
    * Assert:

      * `PriceTrainingResult` is returned.
      * Model can `predict` on a holdout subset.

* **Sub-task 4.3.3 – (Optional) Model registry integration**

  * Extend training pipeline or job (later epic) to:

    * Register the trained model in MLflow Model Registry with name `price_rec_BU1`.
    * Add tags for BU and scenario.

---

## EPIC 5 – Inference & Post-processing (Scenario A)

> Goal: Given new orders or scoring data, produce line-level and deal-level discount ranges with guardrails.

---

### Story 5.1 – Implement guardrail policy for discount bands

**As** a pricing owner
**I want** guardrails on discount recommendations
**So that** the model never recommends extreme or unreasonable discounts.

**Sub-tasks**

* **Sub-task 5.1.1 – Implement `GuardrailPolicy`**

  * In `src/price_desk/model_inference/post_processing/policy.py`:

    * Class `GuardrailPolicy(config: BUConfig)`.
    * Method `apply(self, df: pd.DataFrame) -> pd.DataFrame`:

      * Expects discount quantile columns (e.g., `disc_q_0_10`, `disc_q_0_50`, `disc_q_0_90`).
      * Expects segment key columns from `config.segments.keys`.
      * For Scenario A MVP:

        * Use global BU-level `cap_percentile` and `floor_percentile` as static numeric caps (e.g., 10% and 30% discount).
        * Clip each discount quantile to [floor, cap].
        * Ensure quantile monotonicity after clipping.
      * Optionally add columns like `cap_hit` or `floor_hit`.

* **Sub-task 5.1.2 – Add guardrail tests**

  * `src/tests/test_policy.py`:

    * Construct simple DF with discount quantiles outside allowed range.
    * Run `GuardrailPolicy.apply`.
    * Assert:

      * All discount quantiles are within the configured range.
      * For each row, `disc_q_0_10 ≤ disc_q_0_50 ≤ disc_q_0_90`.

---

### Story 5.2 – Implement deal-level aggregator

**As** an analytics consumer
**I want** deal-level discount ranges
**So that** I can reason about total deal economics, not just individual lines.

**Sub-tasks**

* **Sub-task 5.2.1 – Implement `DealAggregator`**

  * In `src/price_desk/model_inference/post_processing/aggregator.py`:

    * Class `DealAggregator(deal_id_col: str, value_cols=("list_price", "quantity"))`.
    * Method `aggregate(self, df: pd.DataFrame) -> pd.DataFrame`:

      * Expects:

        * `deal_id_col` (for Scenario A, typically `order_id`).
        * `list_price`, `quantity`.
        * Discount quantile columns.
      * Compute `line_value = list_price * quantity`.
      * Group by `deal_id_col`, compute value-weighted average for each quantile:

        * Output columns like `deal_disc_q_0_10`, `deal_disc_q_0_50`, `deal_disc_q_0_90`.

* **Sub-task 5.2.2 – Add aggregator tests**

  * `src/tests/test_aggregator.py`:

    * Single-line deals:

      * Check that deal-level quantiles match line-level quantiles.
    * Multi-line deal:

      * Hand-calc weighted averages, ensure `aggregate()` matches.

---

### Story 5.3 – Implement `PriceInferencePipeline` for Scenario A

**As** an ML engineer
**I want** a reusable inference pipeline
**So that** batch scoring jobs can produce consistent price recommendations.

**Sub-tasks**

* **Sub-task 5.3.1 – Implement inference pipeline**

  * In `src/price_desk/model_inference/price_infer.py`:

    * Class `PriceInferencePipeline(config: BUConfig, model: Regressor)`.
    * Method `run(self, raw_df: FrameAdapter) -> Tuple[pd.DataFrame, pd.DataFrame]`:

      * Use `OrdersFeaturePipeline(config)` to transform `raw_df`.
      * Convert features to Pandas.
      * Call `model.predict_quantiles` with configured quantiles.
      * Join quantiles with features & IDs.
      * Apply `GuardrailPolicy`.
      * Use `DealAggregator` to produce deal-level DF.
      * Return `(line_df, deal_df)`.

* **Sub-task 5.3.2 – Inference pipeline tests**

  * `src/tests/test_price_inference_pipeline.py`:

    * Train a small `GbmQuantileRegressor` on toy data (or use a stub).
    * Create a small PandasFrameAdapter as input.
    * Run `PriceInferencePipeline.run`.
    * Assert:

      * Line DF has discount quantile cols and original IDs.
      * Deal DF has deal IDs and corresponding `deal_disc_q_*` columns.

---

## EPIC 6 – Databricks Jobs & Asset Bundles (Scenario A)

> Goal: Run training and batch inference for BU1 in Databricks using Asset Bundles.

---

### Story 6.1 – Implement Databricks training job entrypoint

**As** a platform engineer
**I want** a job that trains BU1’s price model in Databricks
**So that** monthly retraining is automated.

**Sub-tasks**

* **Sub-task 6.1.1 – Implement `jobs/training_pipeline.py`**

  * Create `src/jobs/training_pipeline.py` with:

    * `main(bu_id: str)` function:

      * Initialize `SparkSession`.
      * Load BU config via `load_bu_config(bu_id)`.
      * Read `config.data_paths.orders_table` into Spark DF.
      * Wrap DF in `SparkFrameAdapter`.
      * Instantiate `PriceTrainingPipeline(config)` and call `fit(adapter)`.
      * (Optional) Register model in MLflow Model Registry with name `price_rec_{bu_id}`.
    * CLI parsing for `--bu_id`.

* **Sub-task 6.1.2 – Manual validation in dev workspace**

  * Run training entrypoint in a Databricks notebook for BU1 with a small dataset.
  * Confirm:

    * MLflow run is created.
    * No runtime errors.

---

### Story 6.2 – Implement Databricks inference job entrypoint

**As** a platform engineer
**I want** a batch inference job for BU1
**So that** recommendations are regularly produced to Delta tables.

**Sub-tasks**

* **Sub-task 6.2.1 – Implement `jobs/inference_pipeline.py`**

  * Create `src/jobs/inference_pipeline.py` with:

    * `main(bu_id: str, model_name: str)`:

      * Initialize `SparkSession`.
      * Load BU config.
      * Read `config.data_paths.scoring_table` or fallback to `orders_table`.
      * Wrap as `SparkFrameAdapter`.
      * Load model from MLflow Model Registry:

        * (For now) use pyfunc or direct artifact as appropriate.
      * Instantiate `PriceInferencePipeline(config, model)`.
      * Run `pipeline.run(adapter)` to get `line_df`, `deal_df`.
      * Convert to Spark DFs.
      * Write to `line_output_table` and `deal_output_table` using Delta.

* **Sub-task 6.2.2 – Manual validation in dev workspace**

  * Run inference entrypoint in Databricks notebook.
  * Confirm:

    * Output tables created.
    * Basic sanity checks on row counts and schemas.

---

### Story 6.3 – Define Asset Bundle resources (jobs & targets)

**As** a platform engineer
**I want** training and inference jobs defined as Bundle resources
**So that** they can be deployed consistently across environments.

**Sub-tasks**

* **Sub-task 6.3.1 – Configure `databricks.yml`**

  * Define targets `dev` and (later) `prod` with:

    * `workspace.host` and `workspace.root_path`.
  * Reference resource YAMLs (below).

* **Sub-task 6.3.2 – Define training job resource**

  * `src/resources/model-workflow-resource.yml`:

    * Define job `price_training_BU1`:

      * Task running `python jobs/training_pipeline.py --bu_id=BU1`.
      * Cluster spec appropriate for training.
      * Schedule: monthly.

* **Sub-task 6.3.3 – Define inference job resource**

  * `src/resources/batch-inference-workflow-resource.yml`:

    * Define job `price_inference_BU1`:

      * Task running `python jobs/inference_pipeline.py --bu_id=BU1 --model_name=price_rec_BU1`.
      * Cluster spec for scoring.
      * Schedule: daily or weekly.

* **Sub-task 6.3.4 – Validate and deploy bundles**

  * Run `databricks bundle validate`.
  * Run `databricks bundle deploy --target=dev`.
  * Confirm jobs appear in Databricks Jobs UI.

---

## EPIC 7 – CI/CD & Quality Gates

> Goal: Automated linting, testing, and bundle validation on each change.

---

### Story 7.1 – Implement CI workflow in GitHub Actions

**As** an engineering lead
**I want** automated checks on each PR
**So that** code quality and tests are enforced.

**Sub-tasks**

* **Sub-task 7.1.1 – Create `.github/workflows/ci.yml`**

  * Steps:

    * Checkout repo.
    * Set up Python.
    * `pip install -r src/requirements.txt`.
    * Lint with `ruff` or `flake8`.
    * Run tests: `pytest src/tests`.
    * (Optional) run `databricks bundle validate` with appropriate credentials.

* **Sub-task 7.1.2 – Validate CI**

  * Open a test PR to trigger CI.
  * Confirm:

    * Lint & tests run.
    * Failures block merge.

---

### Story 7.2 – (Optional) Deploy workflow

**As** a platform engineer
**I want** auto-deploy to dev (and manual to prod)
**So that** changes can be shipped reliably.

**Sub-tasks**

* **Sub-task 7.2.1 – Create `.github/workflows/deploy.yml`**

  * On push to `main`:

    * Run `databricks bundle deploy --target=dev`.
  * Add manual approval step for `prod` (later).

---
