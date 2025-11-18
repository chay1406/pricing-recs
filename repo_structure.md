Price_Desk         <- Root directory. Both monorepo and polyrepo are supported.
│
├── src                  <- Contains all the python code, notebooks, jobs and ML resources related to the ML project 
|   │
│   ├── requirements.txt        <- Specifies Python dependencies for ML code.
|   │
|   ├── databricks.yml          <- databricks.yml is the root bundle file for the ML project that can be loaded by databricks CLI bundles. It defines the bundle name, workspace URL and resource config component to be included. 
|   |
│   ├── price_desk       <- Contains the python code needed to run all the jobs or notebooks.
|   │   │
|   │   ├── data                    <- Optional folder to store any sample data for testing purpose. 
|   │   │
|   │   ├── configs                 <- For parameter configuration files of a pipeline/task/model. 
|   │   │
|   │   ├── utilities               <- For files/code that is common and reusable across different usecases/scenarios (commons.py, feature_utils.py, ml_utils.py, etc.).
|   │   │
|   │   ├── feature_engineering     <- Feature computation code (Python modules) that implements the feature transforms.
|   │   │                              The output of these transforms get persisted as Feature Store tables and gets used for training and validating models. 
|   │   │
|   │   ├── model_deployment        <- For files/code (Python modules) to be used to train/test models, log and register them in unity catalog.
|   │   │   │
|   │   │   ├── models              <- To place the different models to be used/tried for training/testing with OOP wrappers.
|   │   │
|   │   ├── training                <- Training folder contains Notebooks that train and register the model with feature store support.
|   │   │
|   │   ├── model_inference         <- For placing files/code to generate predictions using the registered model and any post-processing scripts.
|   │   │   │
|   │   │   ├── post_proessing      <- Optional sub-folder with code for any post-processing that is required after inference.
|   │   │
|   │   ├── model_monitoring        <- Model monitoring to evaluate if re-training is needed, and feature monitoring to evaluate drifts, etc.
│   │
│   ├── notebooks                   <- For notebooks to use for development and test modules as we go.
│   │
│   ├── tests                   <- Unit tests and integration tests for the entire project.
│   │
│   ├── jobs                    <- For Files/scripts which are starting point/driver of a pipeline execution (feature_pipeline.py, training_pipeline.py, inference_pipeline.py).
│   │
│   ├── resources               <- ML resource (ML jobs, MLflow models) config definitions expressed as code, across dev/staging/prod/test.
│       │
│       ├── model-workflow-resource.yml                <- ML resource config definition for model training, validation, deployment workflow
│       │
│       ├── batch-inference-workflow-resource.yml      <- ML resource config definition for batch inference workflow
│       │
│       ├── feature-engineering-workflow-resource.yml  <- ML resource config definition for feature engineering workflow
│       │
│       ├── monitoring-resource.yml                    <- ML resource config definition for quality monitoring workflow
│
├── .github                     <- Configuration folder for CI/CD using GitHub Actions.  The CI/CD workflows deploy ML resources defined in the `./
├── README.md