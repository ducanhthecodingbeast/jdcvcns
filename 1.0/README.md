# 1.0 independent test project

Goal: run `dotpdtesting1.0.py` as a complete version-1 test project without importing code from `JobCV-matching_CNS` or from `2.0`/`3.0`.

Shared exceptions:
- Data files are read from `../Data`: `jd.csv`, `cv.csv`, `mockcv.csv`.
- Mock CV generation and data preprocessing live in `../Dataset`.

Prepare shared data from the repo root first:

```bash
python3 Dataset/data_preprocessing.py
python3 -m Dataset.mockcv --force
```

Run:

```bash
cd 1.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python3 dotpdtesting1.0.py
```

Core files:
- `dotpdtesting1.0.py`: version-1 test entrypoint.
- `pipeline.py`: version-local dataset/text/embedding helpers.
- `testingresult.py`, `demoAPI/`: version-local PostgreSQL result storage.

Follow-up agents can extend Docker/API integration here, but should not move shared runtime code back outside this folder.
