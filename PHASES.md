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
- `STORE_DB=0`, `WRITE_RESULTS=0`, and `QDRANT_PATH=:memory:` can be passed from the shell and will override `.env.local` for smoke tests.
- `SKIP_PIP_INSTALL=1` skips dependency installation when the environment is already prepared.
