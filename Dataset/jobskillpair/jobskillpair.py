import argparse
import subprocess
import zipfile
from pathlib import Path

DATASET_NAME = "asaniczka/1-3m-linkedin-jobs-and-skills-2024"


def download_dataset(raw_dir: Path, dataset_name: str = DATASET_NAME) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    if list(raw_dir.glob("*.csv")) or list(raw_dir.glob("*.zip")):
        print(f"PASS download: files already exist in {raw_dir}")
        return

    print(f"RUN download: {dataset_name}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(raw_dir)],
        check=True,
    )


def unzip_dataset(raw_dir: Path, remove_zip: bool = True) -> None:
    zip_files = sorted(raw_dir.glob("*.zip"))
    if not zip_files:
        print(f"PASS unzip: no zip files found in {raw_dir}")
        return

    for zip_path in zip_files:
        print(f"RUN unzip: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        if remove_zip:
            zip_path.unlink()


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[2] / "Dataset" / "Data"
    parser = argparse.ArgumentParser(description="Download and unzip the LinkedIn jobs-and-skills dataset.")
    parser.add_argument("--raw-dir", type=Path, default=base_dir / "jobskillpair_raw")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip files after extraction.")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    download_dataset(args.raw_dir, args.dataset)
    unzip_dataset(args.raw_dir, remove_zip=not args.keep_zip)
    print(f"DONE dataset: files are ready in {args.raw_dir}")


if __name__ == "__main__":
    main()
