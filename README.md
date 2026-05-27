# Loan approval predictor

## Running the training

1. Install the package:

```bash
pip install .
```

2. Run the training with MLflow:

```bash
mlflow run . --env-manager <dependency_manager> --experiment-name <experiment_name>
```

- `<dependency_manager>`: `local` to use the current environment, or `uv` / `conda` to let MLflow manage dependencies.
- `<experiment_name>`: name of the MLflow experiment to log runs under (e.g. `loan_approval_predictor`).

Example:

```bash
mlflow run . --env-manager local --experiment-name loan_approval_predictor
```
