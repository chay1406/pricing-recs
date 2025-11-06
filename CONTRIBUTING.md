# Contributing Guidelines — Pricing Recommendations Repo

Welcome!  
This repo follows a structured, multi-phase plan for developing and deploying
the **Price Recommendation & Negotiation AI** across business units.

We use:
- **Codex – OpenAI’s coding agent** (VS Code extension) for in-IDE AI assistance  
- **GitHub Actions** for CI/CD  
- **Databricks Asset Bundles** for orchestration  
- **MLflow** for model tracking & registry  
- **pytest** for validation

---

## 1⃣ Getting Started

### Prerequisites
- Python ≥ 3.10
- macOS or Linux (preferred)
- ChatGPT Plus/Pro account (for Codex sign-in)
- Access to repo: `https://github.com/<your-org>/pricing-recs.git`

### Setup
```bash
git clone https://github.com/<your-org>/pricing-recs.git
cd pricing-recs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt || pip install pytest pydantic mlflow black
pytest -q
```

---

## 2⃣ Branching Model

We follow a simple trunk-based flow:

| Purpose | Branch name pattern | Example |
|----------|--------------------|----------|
| Feature work | `feat/<task-key>-short-desc` | `feat/featurebuilder-v1` |
| Bug fix | `fix/<scope>-short-desc` | `fix/io-spark-adapter` |
| Experiment / spike | `exp/<idea>` | `exp/quantile-baseline` |

Push small PRs frequently; keep diffs ≤ 300 lines.

---

## 3⃣ Working with Codex

Each task from [`CODING_WITH_CODEX.md`](CODING_WITH_CODEX.md)
is run as a **Codex: New Task** in VS Code.

**Before starting a task:**
1. Pull latest main branch.
2. Open VS Code → `Codex: New Task`.
3. Paste the task prompt block (with Constraints).
4. Approve Codex’s plan, apply patches, and run:
   ```bash
   pytest -q
   black --check .
   ```

**After verifying tests pass:**
```bash
git add -A
git commit -m "feat: short summary"
git push origin feat/<task>"
```

Submit a PR to `main` with:
- ✅ Short title (≤ 72 chars)
- ✅ Linked issue / task (if applicable)
- ✅ Green CI checks

---

## 4⃣ Code Style

- Formatter: **Black**
- Imports: **isort** order
- Typing: **PEP 484** type hints required in all public functions
- Docstrings: **Google-style**
- Tests: one per module in `tests/unit/`

---

## 5⃣ Validation

- **Unit tests** → `pytest -q`
- **Integration** → triggered via GitHub Actions (see `.github/workflows/ci.yml`)
- **Bundle validation** → `databricks bundle validate`
- **Model evaluation** → RMSE/MAE + feature importance in MLflow

---

## 6⃣ Review Checklist

Before merging:
- [ ] Code runs locally in pandas mode
- [ ] Tests pass
- [ ] No Spark hard dependencies for pandas path
- [ ] Proper logging (MLflow / console)
- [ ] Config paths relative and valid
- [ ] PR title and description clean

---

## 7⃣ Advanced Workflows

Once base modules are complete:
- Extend to **quantile / conformal ranges**
- Add **pseudo-deal context**
- Integrate **Databricks Bundle** jobs for automation
- Embed in **Salesforce CPQ / Teams chatbot**

---

_Thank you for helping build the Price Desk AI foundation!_
