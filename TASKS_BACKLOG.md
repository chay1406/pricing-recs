# Backlog — Advanced Phases for Pricing Recommendations

This document tracks **future-phase tasks** beyond the initial orders-only pipeline.

Each task can be delegated to Codex via “Codex: New Task” (see CODING_WITH_CODEX.md).

---

## 📦 Phase 3 — Range Models (Quantile / Conformal)

### Task H — Quantile Regression Forest (QRF)
```
Goal: Predict lower, median, and upper discount quantiles per line.

Files:
- src/pricing/models/price_range.py
- notebooks/15_train_quantile.py
- tests/unit/test_model_quantile.py

Details:
- Use GradientBoostingRegressor or RandomForestQuantileRegressor (if available)
- Train three quantiles (0.1, 0.5, 0.9)
- Log to MLflow with tag `model_type=quantile`
- Inference: output columns lower_discount, median_discount, upper_discount
```

### Task I — Conformal Calibration
```
Goal: Calibrate quantile predictions using conformal intervals.

Files:
- src/pricing/models/conformal.py
- tests/unit/test_conformal.py

Details:
- Compute nonconformity scores on holdout set
- Calibrate quantile range to empirical coverage target (90%)
- Add function get_confidence_interval(df, alpha)
```

---

## 🤩 Phase 4 — Pseudo-Deal Context & Win Probability (optional BUs)

### Task J — Pseudo-deal Feature Augmentation
```
Goal: Derive deal-level aggregates for orders lacking quotes.

Files:
- src/pricing/features/deal_context.py
- tests/unit/test_deal_context.py

Details:
- Group by customer/date window (±30 days)
- Add rolling avg_discount, rolling_deal_size, avg_quantity
- Merge pseudo-deal context into line features
```

### Task K — Win Probability Model
```
Goal: Train classifier to predict deal win probability.

Files:
- src/pricing/models/win_prob.py
- notebooks/18_train_winprob.py

Details:
- LogisticRegression or XGBoostClassifier
- Features: deal size, region, segment, discount, past win rate
- Output win_probability; log ROC-AUC, F1
```

---

## ⚙️ Phase 5 — Databricks & Orchestration

### Task L — Databricks Bundle Jobs
```
Goal: Automate training/inference per business unit.

Files:
- .databricks/bundle.yml
- jobs/train_point_job.yml
- jobs/predict_lines_job.yml

Details:
- Define one job per BU with bundle variables (bu_id, config path)
- Schedule monthly retrain
- Use MLflow model registry name pattern pricing_recs_<bu_id>
```

### Task M — CI/CD for Bundles
```
Goal: Add GitHub Actions for Databricks deploy.

Files:
- .github/workflows/deploy.yml

Details:
- Trigger on push to main
- Use databricks bundle validate + deploy
- Separate jobs for dev/stage/prod environments
```

---

## 🤖 Phase 6 — Chatbot & Integration

### Task N — Salesforce/Teams Interface
```
Goal: Build conversational agent wrapper for price recommendations.

Files:
- app/chatbot/price_agent.py
- app/api/serve_model.py

Details:
- Simple FastAPI or Flask app exposing endpoints
- Functions:
  * recommend_price(bu_id, product_id, customer_id, quantity)
  * justify_price(bu_id, product_id) (feature SHAP summary)
- Connect to Salesforce CPQ or Teams bot interface
```

---

## 🢜 Phase 7 — Analytics & Monitoring

### Task O — Price Recommendation Dashboard
```
Goal: Publish Power BI / Databricks SQL dashboard.

Details:
- Visualize recommended vs actual discount distributions
- Track model drift over time
- Include margin impact estimation
```

### Task P — Drift Detection & Retraining Trigger
```
Goal: Monitor data drift and trigger retraining jobs.

Files:
- src/pricing/monitoring/drift.py
- tests/unit/test_drift.py

Details:
- Compare rolling feature distributions to training baseline
- If drift score > threshold, emit MLflow event tag
```

---

## 🧛️ Prioritization
| Priority | Category | Task | Owner |
|-----------|-----------|------|--------|
| P1 | Quantile model | Task H | DS team |
| P1 | Databricks bundle | Task L | MLOps |
| P2 | Pseudo-deal context | Task J | DS |
| P2 | Win prob model | Task K | DS |
| P3 | Chatbot integration | Task N | AI Eng |
| P3 | Drift detection | Task P | MLOps |
| P4 | Dashboards | Task O | BI/Analytics |

---

## ✅ Definition of Done
- Code merged to main
- Tests green on CI
- Databricks job validated (for automation tasks)
- Model registered & metrics visible in MLflow
- Configurable via BU YAML

---

_End of Backlog_
