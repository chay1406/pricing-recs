# Pricing Recs

This repository contains a skeleton implementation of a multi‑tenant pricing recommendation system.  
It is designed to support multiple business units (BUs) from a single code base, while allowing each BU to run its own jobs on Databricks and manage its own models via MLflow.

The project is structured around the Databricks MLOps Stack.  It includes configuration files, job definitions, notebooks, Python modules, and test scaffolding.  Future development will expand upon this skeleton to implement the full modelling workflow described in the project plan.

## Contents

* `.databricks/bundle.yml` — Sample Databricks Asset Bundle definition for deploying jobs.
* `.github/workflows/` — Placeholder CI workflow definitions.  These will be expanded to lint, test, and deploy the codebase using GitHub Actions.
* `configs/` — Example configuration files for the default BU and a sample BU (BU1).  These files define data locations, execution modes, and basic model settings.
* `notebooks/` — Thin orchestration notebooks used by the Databricks jobs.  Each notebook delegates work to functions in the `src/pricing` package.
* `src/pricing/` — The Python package containing I/O connectors, feature engineering, models, inference logic, and MLOps utilities.  The modules here provide minimal class definitions and are intended to be extended.
* `tests/` — Placeholder unit tests to establish a test suite.  Tests will be added as functionality is implemented.

## Getting Started

To develop locally:

1. Install the dependencies from `pyproject.toml` or add your own to a virtual environment.
2. Update the configuration files under `configs/` for your specific BU.
3. Execute the notebooks from the `notebooks/` directory or run the Python modules directly.
4. Run the test suite with `pytest` to ensure everything is wired correctly.

This skeleton is intentionally lightweight and does not implement any real pricing logic yet.  It provides a foundation upon which agents and developers can collaborate in parallel, following the plan described in the project documentation.
borate in parallel, following the plan described in the project documentation.
