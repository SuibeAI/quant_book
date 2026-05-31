from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile


DATA_URL = (
	"https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-"
	"Neural-Networks-for-Limit-Order-Books/master/data/data.zip"
)
DEFAULT_OUTPUT_DIR = Path.cwd() / "data"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download FI-2010 data for DeepLOB.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where the txt files will be stored.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Re-download and overwrite existing files.",
	)
	return parser.parse_args()


def download_zip(zip_path: Path) -> None:
	zip_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"Downloading data to: {zip_path}")
	urllib.request.urlretrieve(DATA_URL, zip_path)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	with ZipFile(zip_path) as archive:
		archive.extractall(output_dir)


def flatten_nested_data_dir(output_dir: Path) -> None:
	nested_dir = output_dir / "data"
	if not nested_dir.is_dir():
		return

	for item in nested_dir.iterdir():
		target = output_dir / item.name
		if target.exists():
			if target.is_dir():
				shutil.rmtree(target)
			else:
				target.unlink()
		shutil.move(str(item), str(target))

	shutil.rmtree(nested_dir, ignore_errors=True)


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir.resolve()

	if output_dir.is_dir() and not args.force:
		print("data/ already exists, skip download.")
		return

	zip_path = output_dir / "data.zip"
	download_zip(zip_path)
	extract_zip(zip_path, output_dir)
	flatten_nested_data_dir(output_dir)

	print("Data prepared in data/.")


if __name__ == "__main__":
	main()
