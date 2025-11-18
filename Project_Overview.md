# Price Desk AI – Problem Overview, Objectives, Scenarios, and Phased Plan

## 1. Business Problem & Context

The company is a multi-BU industrial / manufacturing business operating in **B2B** markets. Each **Business Unit (BU)**:

* Sells products (and some services) into different industries/regions.
* Uses **SAP** for orders/bookings.
* Uses **Salesforce** (often with CPQ) for opportunities/quotes, but **data quality and adoption vary widely** across BUs.

### Core Problem

Sellers and price desk teams need **data-driven discount / price recommendations** to:

* Negotiate deals more effectively,
* Protect margins,
* Move faster while still staying within business rules.

However:

* Data is fragmented and inconsistent across BUs (especially on quotes and win/loss).
* Many transactions never go through quotes (direct orders), even though they may involve negotiation.
* For some BUs, quotes cannot be reliably linked to SAP orders.

### High-Level Objective

Build a **single, generalized codebase** that:

* Trains BU-specific models using **Databricks** (Spark + MLflow),
* Produces **line-level and deal-level discount ranges** (not just point estimates),
* Logs and serves recommendations via **Delta tables** (Power BI first, API later),
* Supports multiple **data availability scenarios** across BUs,
* Integrates with **Databricks Asset Bundles** for job orchestration and **GitHub Actions** for CI/CD,
* Is **modular and OOP**, enabling multiple developers/agents to work in parallel.

---

## 2. Modeling Objective

For each deal / quote (or inferred “deal” from orders):

1. Generate **line-level discount recommendations**:

   * As **ranges**: e.g., P10/P50/P90 discount bands.
   * With guardrails and justification.

2. Aggregate line-level recommendations to the **deal level**:

   * Produce a **deal-level discount band** (again P10/P50/P90).
   * Ensure the line prices and deal totals are consistent.

3. (In richer scenarios) incorporate:

   * **Win-probability** modeling (where win/loss data exists),
   * **Negotiation regime** modeling (run-rate vs negotiated),
   * More advanced uncertainty modeling and optimization.

Initial **evaluation** is business-driven:

* Show that historical recommendations would have been **reasonable** for previous periods,
* Provide intuitive **justifications** (similar past deals, key drivers),
* Enable BU-specific dashboards where each BU only sees its own data.

---

## 3. Data Availability Scenarios

Different BUs fall into different scenarios depending on the data they have and how cleanly systems are integrated.

### Scenario A – Orders-only (no usable quotes / win-lost)

* **Available:**

  * SAP **orders/bookings** data (final prices, discounts, etc.).
* **Not available / unreliable:**

  * Quotes,
  * Win/loss at line or deal level,
  * Quote→order mapping.

**Implications:**

* Orders are a **mixture** of run-rate and negotiated deals, but we **cannot label them separately**.
* Only sensible option is a **single regression model** trained on all orders.
* No proper win-probability model; no supervised gate between run-rate and negotiated.

---

### Scenario B – Orders + quotes + clean mapping + win/loss

* **Available:**

  * SAP orders,
  * Salesforce quotes with **win/loss labels**,
  * Reliable **quote↔order mapping**,
  * Possibly process metadata (approvals, revisions, etc.).

**Implications:**

* Can train:

  * A **price model** (regression) with richer features (orders + quotes),
  * A **win-probability model** (logistic) using won/lost quotes,
  * A **gate** / negotiation propensity model (run-rate vs negotiation).
* Enables **mixture-of-experts** model and more advanced scenario planning (price vs win vs margin).

---

### Scenario C – Orders + quotes (wins only, partial mapping)

* **Available:**

  * Orders,
  * Quotes mostly for **won** deals, few/no lost quotes.
* **Missing:**

  * Robust win/loss labels (heavily skewed),
  * Reliable mapping for all quotes.

**Implications:**

* Win-probability model is weak or not viable (heavily biased).
* Price model should still be trained on **all orders**, not just quotes.
* If quotes are used for a sub-model, we need **selection-bias correction** (e.g., IPS/Heckman).
* Gate may not be well-supervised; we treat negotiated vs run-rate mostly as a blended population.

---

### Scenario D – Orders + quotes with wins & losses, but poor mapping

* **Available:**

  * Orders,
  * Quotes with win/loss,
* **Issue:**

  * Quotes cannot be reliably tied back to orders.

**Implications:**

* Train **price model** on orders only.
* Train **win-prob model** on quotes only, and use it in a **decoupled** fashion:

  * At inference, simulate deal prices and map them to P(win) using quote-based model,
  * But **do not** attempt joint training with orders.
* Gate can be used if there is sufficient metadata (e.g., quote type) to label negotiated deals.

---

### Scenario E – Mixed quality within a single BU

* One BU has **heterogeneous data quality**:

  * Some lines / product families have clean quotes & mapping,
  * Others look like Scenario A or C.

**Implications:**

* Use a **mixture-of-experts** approach:

  * Head A: run-rate model for orders-only segments,
  * Head B: negotiated model where good quote data exists,
  * Gate model decides which head to trust for a given deal.
* Still use **common infrastructure** and config; only models and features vary per segment.

---

## 4. Core Modeling Concepts

### 4.1 Line-level modeling

* Model targets: **discount_pct** (or equivalently price/list).

* Features include:

  * Product attributes (family, category),
  * Customer attributes (tier, country/region),
  * Transaction attributes (qty, order date),
  * Pseudo deal context (**Scenario A**):

    * customer-week total value,
    * number of lines per customer-week,
    * etc.

* Result: **line-level quantile predictions**:

  * e.g., `disc_q_0_10`, `disc_q_0_50`, `disc_q_0_90`.

### 4.2 Deal-level aggregation

* Quotes / deals often include **multiple lines**.
* For a given deal, compute deal-level discount quantiles as **value-weighted averages** of line-level quantiles (MVP).
* Later:

  * Use **simulation / copula** methods to aggregate full predictive distributions,
  * Optional optimization (QP) to derive consistent line prices that match a chosen deal quantile.

### 4.3 Segments & segment keys

* **Segment keys**: business-defined fields used for:

  * Guardrails,
  * Calibration,
  * Monitoring.
* Examples:

  * `product_family, region, customer_tier, qty_band`.
* Segments are not the same as model leaves; they provide **stable, interpretable buckets**.

### 4.4 Discount ranges (bands)

* Instead of single point predictions, the model gives **ranges**:

  * P10/P50/P90 discount (or more quantiles).
* Guardrail + calibration layer then:

  * Clips bands within reasonable bounds per segment,
  * Ensures quantiles remain monotonic,
  * Optionally calibrates bands to historical distributions.

---

## 5. Phased Implementation Plan

We implement incrementally, starting with the **simplest realistic scenario** and building towards more complex ones.

### Phase 0 – Foundations (shared for all BUs/scenarios)

**Objectives:**

* Establish core infrastructure:

  * Logging,
  * Config system (BU-specific),
  * DataFrame abstraction (Spark/Pandas),
  * MLflow helpers,
  * Repository structure aligned with Databricks Asset Bundles and GitHub Actions.

**Key components:**

* `get_logger` for consistent logging.
* `FrameAdapter` with `PandasFrameAdapter` & `SparkFrameAdapter`.
* Pydantic configs (`BUConfig`, `DataPaths`, `GuardrailConfig`, etc.).
* MLflow helper functions and bundle / CI skeletons.

---

### Phase 1 – Scenario A Vertical Slice (Orders-only, BU1)

**Objective:**
Deliver an end-to-end pipeline for one BU using only **orders data**, with:

* Feature engineering,
* Price model training (regression with quantiles),
* Guardrail policy,
* Line→deal aggregation,
* Batch inference writing **Delta tables** for Power BI,
* MLflow tracking and Databricks jobs via Asset Bundles.

**Key elements:**

1. **Feature Engineering**

   * `OrdersFeaturePipeline`:

     * Compute core derived variables (`discount_pct`, `qty_band`, pseudo deal context),
     * Ensure all segment keys exist.

2. **Model Training**

   * `GbmQuantileRegressor`:

     * LightGBM (or similar) with:

       * one model for point prediction,
       * separate quantile models for P10/P50/P90.
   * `PriceTrainingPipeline`:

     * Orders → features → Pandas → train → log to MLflow.

3. **Post-processing**

   * `GuardrailPolicy`:

     * Per-BU (and later per-segment) caps/floors on discount bands.
   * `DealAggregator`:

     * Value-weighted line→deal aggregation of discount bands.

4. **Inference**

   * `PriceInferencePipeline`:

     * Raw orders / scoring set → features → model quantiles → guardrails → deal aggregation → line & deal outputs.

5. **Jobs & Outputs**

   * Databricks jobs:

     * Training job per BU (e.g., monthly).
     * Batch inference job per BU (daily/weekly).
   * Outputs:

     * `price_recs_line` (line-level bands),
     * `price_recs_deal` (deal-level bands).

**Result:**
A working system for BU1 that produces **discount ranges** from orders data alone.

---

### Phase 2 – Multi-BU Rollout for Scenario A

**Objective:**
Extend Phase 1 architecture to more BUs that are in Scenario A (orders-only or effectively orders-only).

**Activities:**

* Add new BU config files (`bu_BU2.yaml`, etc.).
* Add or parameterize jobs for each BU.
* Ensure any BU-specific nuances are handled by config, not code fork.

---

### Phase 3 – Scenarios C/D – Adding Negotiation Signal & Win-Prob (Decoupled)

**Objective:**
For BUs with quotes but with limitations (wins-only or poor mapping), introduce:

* A **negotiation propensity / gate** where possible (if sufficient metadata),
* A **win-probability model** trained on quotes (kept separate from orders),
* Use win-prob estimates to **annotate** the discount bands, not to redefine the core price model.

**Key additions:**

* `negotiation_features.py` (quote/approval/process features).
* `gate_classifier.py` (logistic/XGB).
* `sk_logistic.py` (win-prob model).
* Augmented diagnostics & dashboards (e.g., show P(win) vs discount curve).

**Important constraint:**
In Scenario A, we **do not** have reliable labels for run-rate vs negotiated, so we do **not** train a gate there. The gate only appears where negotiation signal is observable (quotes/process metadata).

---

### Phase 4 – Scenario B/E – Full Power (Mixture-of-Experts, Simulation)

**Objective:**
For BUs with strong CPQ data and mapping, and for mixed-quality BUs where some segments have great data:

* Introduce **mixture-of-experts**:

  * Run-rate head vs negotiated head,
  * Gate model decides mixture.
* Upgrade deal-level bands using **Monte Carlo / copula-based aggregation**.
* Optionally optimize line prices to match target deal quantiles with constraints.

**Key additions:**

* `mixture.py` (mixture-of-experts implementation).
* More advanced `aggregator.py`:

  * Residual correlation estimation,
  * Gaussian copula sampling,
  * MonteCarlo-based deal-level ranges.
* Optional `optimizer.py` (for QP-based consistent line prices).

---

### Phase 5 – Monitoring, Feedback, and Continuous Improvement

**Objective:**
Operationalize and improve over time.

**Activities:**

* Monitoring jobs:

  * Data drift per segment & BU,
  * Band coverage vs actuals,
  * Guardrail hit rates,
  * Adoption metrics where possible.
* Feedback loop:

  * Use seller/BU feedback to adjust guardrails, segments, and feature engineering.
  * Iterate on win-prob and negotiation models as more labels and cleaner mappings become available.

---

## 6. Summary

* The **short-term focus** is a robust, scalable implementation of **Scenario A** (orders-only) for **one BU**, producing meaningful discount ranges and deal-level bands using orders data.
* The **architecture is deliberately modular and OOP**:

  * Shared abstractions (`FrameAdapter`, `FeaturePipeline`, `Regressor`) allow reuse across scenarios and BUs.
  * Config-driven segments and guardrails give business control and interpretability.
* Later phases **layer in complexity** as data allows:

  * negotiation detection (gate),
  * win-probability modeling,
  * mixture-of-experts,
  * advanced uncertainty aggregation and optimization.

This staged approach lets you **prove value quickly** for one BU while building an architecture that can grow into a full **Price Desk AI platform** across all BUs and data scenarios.
