from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = PROJECT_ROOT / "pipeline.yml"


def load_config() -> dict[str, Any]:
    with PIPELINE_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def path_exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def dataset_ready(config: dict[str, Any]) -> bool:
    required = config["dataset"]["required_files"]
    jd_ready = path_exists(required["jd"])
    cv_ready = any(path_exists(path) for path in required["cv"])
    return jd_ready and cv_ready


def run_command(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_scripts(scripts: list[str], env: dict[str, str], max_workers: int) -> list[dict[str, Any]]:
    commands = [[sys.executable, script] for script in scripts]
    if max_workers <= 1:
        return [run_command(command, env) for command in commands]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_command, command, env) for command in commands]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JobCV matching server pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Do not run dataset preprocessing.")
    parser.add_argument("--force-preprocess", action="store_true", help="Run preprocessing even if normalized CSVs exist.")
    parser.add_argument("--include-heavy", action="store_true", help="Also run the Qdrant/BGE-M3 experiment.")
    parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K for experiment scripts.")
    parser.add_argument("--no-db", action="store_true", help="Skip writing results to Postgres.")
    parser.add_argument("--max-workers", type=int, default=1, help="Run experiments concurrently when greater than 1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    started = time.monotonic()

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in config.get("environment", {}).items()})
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["STORE_DB"] = "0" if args.no_db else env.get("STORE_DB", "1")
    if args.top_k is not None:
        env["TOP_K"] = str(max(1, args.top_k))

    results: list[dict[str, Any]] = []

    if not args.skip_download and (args.force_preprocess or not dataset_ready(config)):
        preprocess_scripts = config["dataset"]["preprocessing_scripts"]
        preprocess_script = next((script for script in preprocess_scripts if path_exists(script)), None)
        if preprocess_script is None:
            raise FileNotFoundError("No preprocessing script from pipeline.yml exists.")
        results.append(run_command([sys.executable, preprocess_script], env))

    default_scripts = [script for script in config["scripts"]["default"] if path_exists(script)]
    results.extend(run_scripts(default_scripts, env, max_workers=max(1, args.max_workers)))

    if args.include_heavy:
        heavy_scripts = [script for script in config["scripts"]["heavy"] if path_exists(script)]
        results.extend(run_scripts(heavy_scripts, env, max_workers=1))

    failed = [result for result in results if result["returncode"] != 0]
    summary = {
        "script_count": len(results),
        "failed_count": len(failed),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
