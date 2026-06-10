# Demo API (Results Dashboard)

FastAPI service that reads/writes **testing results** from Postgres (pgvector image is fine).

## Run (local, without Docker)

Set env vars (defaults match `Docker/compose.yml`):

- `DATABASE_URL=postgresql://jdcvcns:jdcvcns_dev_password@localhost:5432/jdcvcns`

Then:

```bash
python -m pip install -r requirements.api.txt
uvicorn demoAPI.main:app --reload --port 8000
```

Open:
- `http://localhost:8000/` (serves the frontend)
- `http://localhost:8000/docs` (API docs)

## Writing results from scripts

All updated test scripts default to:
- `STORE_DB=1` (write to Postgres)

## Server run order

From the repo root on the server:

```bash
cd Docker
docker compose up --build
```

In another shell, prepare data if `Dataset/jd.csv` and `Dataset/cv.csv` are missing:

```bash
python "Dataset/data preprocessing.py"
```

The Kaggle dataset is `phamtheds/job-dataset-for-recommendation`:

```text
https://www.kaggle.com/datasets/phamtheds/job-dataset-for-recommendation
```

For non-interactive servers, set `KAGGLE_USERNAME` and `KAGGLE_KEY` before running the preprocessing script.

Then run the experiments:

```bash
python "1.0/dotpdtesting1.0.py"
python "2.0/dotpdtesting2.1.py"
python "2.0/dotpdtesting2.2.py"
python "2.0/dotpdtesting2.3.py"
```

Or run the full server pipeline:

```bash
python run_server_pipeline.py
```

Open the dashboard at `http://localhost:8000/`.
