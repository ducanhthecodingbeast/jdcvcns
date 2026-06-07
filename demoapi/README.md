# Demo API (Results Dashboard)

FastAPI service that reads/writes **testing results** from Postgres (pgvector image is fine).

## Run (local, without Docker)

Set env vars (defaults match `Practice and Research/Docker/compose.yml`):

- `DATABASE_URL=postgresql://dangnguyenducanh:postgres@localhost:15566/jdcvmatching`

Then:

```bash
python -m pip install -r "Practice and Research/Too lazy to care abt/requirements.txt"
uvicorn demo_api.main:app --reload --port 8000
```

Open:
- `http://localhost:8000/` (serves the frontend)
- `http://localhost:8000/docs` (API docs)

## Writing results from scripts

All updated test scripts default to:
- `STORE_DB=1` (write to Postgres)
- `EXPORT_HTML=0` (skip heavy HTML)

To force HTML export:

```bash
set EXPORT_HTML=1
python "Practice and Research/dotpdtesting2.1.py"
```

## Server run order

From the repo root on the server:

```bash
cd "Practice and Research/Docker"
docker compose up --build
```

In another shell, prepare data if `Dataset/jd.csv` and `Dataset/cv.csv` are missing:

```bash
cd "Practice and Research"
python "data preparing/data preprocessing.py"
    similarities = jd_embeddings @ cv_embeddings.T

```

The Kaggle dataset is `phamtheds/job-dataset-for-recommendation`:

```text
https://www.kaggle.com/datasets/phamtheds/job-dataset-for-recommendation
```

For non-interactive servers, set `KAGGLE_USERNAME` and `KAGGLE_KEY` before running the preprocessing script.

Then run the experiments:

```bash
python dotpdtesting1.0.py
python dotpdtesting2.1.py
python dotpdtesting2.2.py
python dotpdtesting2.3.py
```

Or run the full server pipeline:

```bash
python run_server_pipeline.py
```

Open the dashboard at `http://localhost:8000/`.
