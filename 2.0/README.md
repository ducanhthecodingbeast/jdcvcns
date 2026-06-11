# 2.0 independent test project

Goal: run every `dotpdtesting2.x.py` file as part of a complete version-2 test project without importing code from `JobCV-matching_CNS` or from `1.0`/`3.0`.

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
cd 2.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python3 dotpdtesting2.1.py
python3 dotpdtesting2.2.py
python3 dotpdtesting2.3.py
```

Run fully in Docker Compose:

```bash
cd 2.0
docker compose run --rm test
TEST_FILE=dotpdtesting2.1.py docker compose run --rm test
TEST_FILE=dotpdtesting2.2.py docker compose run --rm test
TEST_FILE=dotpdtesting2.3.py docker compose run --rm test
```

`dotpdtesting2.0.py` uploads JD embeddings to PostgreSQL/pgvector. The `2.1`, `2.2`, and `2.3` files run matching tests and store result runs.

Follow-up agents can improve result export/reporting, but should keep shared runtime code version-local inside this folder.
