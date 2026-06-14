# Resumable Benchmark Phases

Use this workflow for the long CV/JD ranking runs so a crash in a later phase does not force earlier phases to run again.

## Default Server Run

```bash
./run_all.sh
```

The default phase order is:

```text
data -> 1.0 -> 2.1 -> 2.2 -> 2.3 -> 4.0 -> 3.0 -> 6.0 -> 6.1 -> 6.2
```

Each successful phase writes a marker under:

```text
.phase_state/default/<phase>.done
```

If a phase fails, fix the issue and rerun the same command. Completed phases are skipped automatically.

## Project Port Map

Do not point this project at default/common server ports such as PostgreSQL
`5432`, Qdrant `6333`, Pinecone Local `5080`, Ollama `11434`, API `8000`, or
Vite `5173`. Use separate external subports so this project does not conflict
with other projects on the same server.

Host-facing defaults used by this repo:

```text
1.0 PostgreSQL:      15410
2.0 PostgreSQL:      15420
3.0 PostgreSQL:      15430
4.0 PostgreSQL:      15440
6.x PostgreSQL:      15600
Pinecone Local:      15080-15090
Dataset Ollama URL:  16434
Demo backend API:    18000
Demo frontend:       15173
```

Optional legacy Qdrant override ports are `16330` for 3.0 and `16340` for 4.0.
They are not started by default; set `VECTOR_BACKEND=qdrant` explicitly to use them.

Server DB examples:

```bash
export STORE_DB=1
export WRITE_RESULTS=0

DATABASE_URL='postgresql://user:pass@server-host:15430/jdcvcns_30' ./3.0/run.sh
DATABASE_URL='postgresql://user:pass@server-host:15440/jdcvcns_40' ./4.0/run.sh
DATABASE_URL='postgresql://user:pass@server-host:15600/jdcvcns_60' ./6.0/run.sh all
```

For Demo, point each source at the same non-default server ports:

```bash
export DEMO_DATABASE_URL_30='postgresql://user:pass@server-host:15430/jdcvcns_30'
export DEMO_DATABASE_URL_40='postgresql://user:pass@server-host:15440/jdcvcns_40'
export DEMO_DATABASE_URL_60='postgresql://user:pass@server-host:15600/jdcvcns_60'
```

The `:5432`, `:6333`, `:5080`, and application framework ports that still
appear in Dockerfiles or `compose.yaml` are container-internal ports required by
their images/processes. Host/server access stays on the mapped subports above.

## Background Run

```bash
./run_all.sh --background
tail -f .phase_state/default/run.log
```

Per-phase logs are stored under:

```text
.phase_state/default/logs/
```

## Useful Commands

List phases:

```bash
./scripts/phased_suite.py --list
```

Use a named run:

```bash
./scripts/phased_suite.py --run-id server-a
```

Retry only one phase:

```bash
./scripts/phased_suite.py --run-id server-a 4.0
```

Run a phase group:

```bash
./scripts/phased_suite.py --run-id server-a 2.0
./scripts/phased_suite.py --run-id server-a 6
```

Force a completed phase to rerun:

```bash
./scripts/phased_suite.py --run-id server-a --force 4.0
```

Reset all markers for a run id:

```bash
./scripts/phased_suite.py --run-id server-a --reset
```

## Notes

- `2.0/run.sh` defaults to the ranking variants `2.1`, `2.2`, and `2.3`, but the phased suite checkpoints those variants separately.
- `6.0` phase names map to scripts as: `6.0` cosine, `6.1` dot product, `6.2` BM25.
- `Dataset/run.sh --check-only` validates existing `Data/jd.csv`, `Data/cv.csv`, and `Data/mockcv.csv` without Kaggle or local LLM generation.
- `STORE_DB=0` and `WRITE_RESULTS=0` can be passed from the shell and will override `.env.local` for smoke tests.
- `SKIP_PIP_INSTALL=1` skips dependency installation when the environment is already prepared.

## Pinecone Local

Pinecone Local is the default vector backend for the vector phases:

```bash
export PINECONE_HOST=http://localhost:15080
./run_all.sh --background
tail -f .phase_state/default/run.log
```

This starts a shared in-memory Docker container named `jdcvcns-pinecone-local`.
Pinecone Local does not persist records after the container is removed.
