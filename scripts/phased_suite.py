#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORDER = ["data", "1.0", "2.1", "2.2", "2.3", "4.0", "3.0", "6.0", "6.1", "6.2"]


@dataclass(frozen=True)
class Phase:
    name: str
    command: list[str]
    description: str


PHASES: dict[str, Phase] = {
    "data": Phase(
        name="data",
        command=["./Dataset/run.sh", "--check-only"],
        description="Validate Data/jd.csv, Data/cv.csv, and reusable Data/mockcv.csv.",
    ),
    "1.0": Phase(
        name="1.0",
        command=["./1.0/run.sh"],
        description="Run version 1.0 CV-to-JD dot-product ranking.",
    ),
    "2.1": Phase(
        name="2.1",
        command=["./2.0/run.sh", "2.1"],
        description="Run version 2.1 dot-product CV-to-JD ranking.",
    ),
    "2.2": Phase(
        name="2.2",
        command=["./2.0/run.sh", "2.2"],
        description="Run version 2.2 two-phase hybrid CV-to-JD ranking.",
    ),
    "2.3": Phase(
        name="2.3",
        command=["./2.0/run.sh", "2.3"],
        description="Run version 2.3 direct BGE-M3 hybrid CV-to-JD ranking.",
    ),
    "4.0": Phase(
        name="4.0",
        command=["./4.0/run.sh"],
        description="Run version 4.0 Qdrant hybrid retrieval with ViRanker reranking.",
    ),
    "3.0": Phase(
        name="3.0",
        command=["./3.0/run.sh"],
        description="Run version 3.0 BGE-M3 hybrid Qdrant ranking.",
    ),
    "6.0": Phase(
        name="6.0",
        command=["./6.0/run.sh", "6.0"],
        description="Run version 6.0 JobBERT cosine ranking.",
    ),
    "6.1": Phase(
        name="6.1",
        command=["./6.0/run.sh", "6.1"],
        description="Run version 6.1 JobBERT dot-product ranking.",
    ),
    "6.2": Phase(
        name="6.2",
        command=["./6.0/run.sh", "6.2"],
        description="Run version 6.2 BM25 lexical ranking.",
    ),
}


GROUPS = {
    "all": DEFAULT_ORDER,
    "2": ["2.1", "2.2", "2.3"],
    "2.0": ["2.1", "2.2", "2.3"],
    "2.x": ["2.1", "2.2", "2.3"],
    "6": ["6.0", "6.1", "6.2"],
    "6.x": ["6.0", "6.1", "6.2"],
}

ALIASES = {
    "1": "1.0",
    "3": "3.0",
    "4": "4.0",
    "dataset": "data",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the long CV/JD benchmark suite as resumable phases. "
            "Completed phases are skipped on the next run unless --force or --reset is used."
        )
    )
    parser.add_argument(
        "phases",
        nargs="*",
        help="Phase names/groups to run. Default: data 1.0 2.1 2.2 2.3 4.0 3.0 6.0 6.1 6.2.",
    )
    parser.add_argument("--run-id", default=os.environ.get("PHASE_RUN_ID", "default"))
    parser.add_argument("--state-dir", default=os.environ.get("PHASE_STATE_DIR", str(REPO_ROOT / ".phase_state")))
    parser.add_argument("--reset", action="store_true", help="Delete state for this run id before running.")
    parser.add_argument("--force", action="store_true", help="Run phases even if their .done marker exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later phases after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned phases without executing them.")
    parser.add_argument("--list", action="store_true", help="List available phases and exit.")
    return parser.parse_args()


def expand_phase_name(name: str) -> list[str]:
    normalized = ALIASES.get(name, name)
    return GROUPS.get(normalized, [normalized])


def selected_phases(names: list[str]) -> list[Phase]:
    selected: list[str] = []
    for name in names:
        selected.extend(expand_phase_name(name))
    if not selected:
        selected = list(DEFAULT_ORDER)

    unknown = [name for name in selected if name not in PHASES]
    if unknown:
        available = ", ".join(DEFAULT_ORDER + sorted(GROUPS))
        raise SystemExit(f"Unknown phase(s): {', '.join(unknown)}. Available phases/groups: {available}")
    return [PHASES[name] for name in selected]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def marker_path(run_dir: Path, phase: Phase) -> Path:
    return run_dir / f"{phase.name}.done"


def log_path(run_dir: Path, phase: Phase) -> Path:
    return run_dir / "logs" / f"{phase.name}.log"


def write_marker(run_dir: Path, phase: Phase, elapsed_seconds: float) -> None:
    path = marker_path(run_dir, phase)
    payload = {
        "phase": phase.name,
        "description": phase.description,
        "command": phase.command,
        "completed_at": now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_phase(run_dir: Path, phase: Phase) -> int:
    path = log_path(run_dir, phase)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase {phase.name}: {phase.description} ===")
    print("Command:", " ".join(phase.command))
    print("Log:", path)

    started = datetime.now(timezone.utc)
    with path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"phase={phase.name}\n")
        log_file.write(f"description={phase.description}\n")
        log_file.write(f"command={' '.join(phase.command)}\n")
        log_file.write(f"started_at={started.isoformat()}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            phase.command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        code = process.wait()
        finished = datetime.now(timezone.utc)
        elapsed = (finished - started).total_seconds()
        log_file.write(f"\nexit_status={code}\n")
        log_file.write(f"finished_at={finished.isoformat()}\n")
        log_file.write(f"elapsed_seconds={elapsed:.3f}\n")

    if code == 0:
        write_marker(run_dir, phase, elapsed)
        print(f"Phase {phase.name} completed.")
    else:
        print(f"Phase {phase.name} failed with exit status {code}. Fix the issue and rerun this command to resume.")
    return code


def main() -> int:
    args = parse_args()
    if args.list:
        for name in DEFAULT_ORDER:
            phase = PHASES[name]
            print(f"{phase.name}: {phase.description}")
        return 0

    phases = selected_phases(args.phases)
    state_dir = Path(args.state_dir).resolve()
    run_dir = state_dir / args.run_id

    if args.reset and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Run id:", args.run_id)
    print("State:", run_dir)
    print("Phases:", " ".join(phase.name for phase in phases))

    failed: list[str] = []
    for phase in phases:
        marker = marker_path(run_dir, phase)
        if marker.exists() and not args.force:
            print(f"Skipping phase {phase.name}; marker exists at {marker}")
            continue
        if args.dry_run:
            print(f"Would run phase {phase.name}: {' '.join(phase.command)}")
            continue

        code = run_phase(run_dir, phase)
        if code != 0:
            failed.append(phase.name)
            if not args.continue_on_error:
                return code

    if failed:
        print("Failed phases:", " ".join(failed), file=sys.stderr)
        return 1
    print("\nAll requested phases completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
