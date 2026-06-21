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

## Serving the model as a REST API

Once a model has been logged, you can serve it as a REST API with `mlflow models serve`.

> **Requirement:** `pyenv` must be installed so MLflow can provision the Python version required by the model. On Windows, install [`pyenv-win`](https://github.com/pyenv-win/pyenv-win).

1. Find the run ID of the model you want to serve (visible in the MLflow UI or under `mlruns/`).

2. Start the server, pointing at the logged model and choosing a port:

```bash
mlflow models serve -m runs:/<run_id>/loan_approval_model --port <port> --env-manager local
```

- `<run_id>`: ID of the run that logged the model (e.g. `e881406e7ca54fddaff0b9d62a69e832`).
- `<port>`: port to expose the API on (e.g. `1234`).

Example:

```bash
mlflow models serve -m runs:/e881406e7ca54fddaff0b9d62a69e832/loan_approval_model --port 1234
```

### Making a prediction

Send a `POST` request to the `/invocations` endpoint with the input records as JSON. Each record must contain the model's expected features.

On Windows (PowerShell), using `curl.exe`:

```powershell
curl.exe -X POST http://127.0.0.1:1234/invocations `
  -H "Content-Type: application/json" `
  -d '{\"dataframe_records\": [{\"education\": \"Graduate\", \"self_employed\": \"No\", \"no_of_dependents\": 2, \"income_annum\": 9600000, \"loan_amount\": 29900000, \"loan_term\": 12, \"cibil_score\": 778, \"luxury_assets_value\": 22700000, \"residential_assets_value\": 2400000, \"commercial_assets_value\": 17600000, \"bank_asset_value\": 8000000}]}'
```

The server responds with the model's prediction, for example:

```json
{"predictions": ["Approved"]} 
```

## Connecting to MySQL database

Instead of the default SQLite backend, you can use MySQL to store MLflow tracking data.

1. Install [MySQL](https://dev.mysql.com/downloads/installer/) and create a database and user:

```sql
CREATE USER 'loan_user'@'localhost' IDENTIFIED BY 'password';
CREATE DATABASE loan_db;
GRANT ALL PRIVILEGES ON loan_db.* TO 'loan_user'@'localhost';
FLUSH PRIVILEGES;
```

2. Start the MLflow server pointing at the MySQL backend:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri mysql+pymysql://loan_user:password@localhost/loan_db \
  --default-artifact-root ./mlruns
```

3. Set the tracking URI so your training runs are sent to this server:

```bash
# PowerShell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"

# Linux / macOS
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

From here, run training and serve the model as described in the sections above. To serve a registered model version or alias instead of a run artifact, replace the `-m` argument:

```bash
# By version number
mlflow models serve -m models:/loan_approval_model/1 --port <port> --env-manager local

# By alias (e.g. champion, staging)
mlflow models serve -m models:/loan_approval_model@champion --port <port> --env-manager local
```
