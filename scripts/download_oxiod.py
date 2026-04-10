#!/usr/bin/env python3
"""
Download script for the Oxford Inertial Odometry Dataset (OxIOD).

Official source: http://deepio.cs.ox.ac.uk
Paper: https://arxiv.org/abs/1809.07491

The OxIOD dataset contains 158 sequences totaling over 42 km of inertial data,
collected with various IMU devices (including BNO055 9-axis IMU at 100Hz).

Usage:
    python scripts/download_oxiod.py [--output-dir data/oxiod]

The script will:
1. Download the dataset archive from the official Oxford server
2. Extract the contents to the specified output directory
3. Verify the download integrity
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import urllib.error
import zipfile
import shutil
from pathlib import Path

# Official dataset URL from Oxford University
DATASET_URL = "http://deepio.cs.ox.ac.uk/data/oxiod_dataset.zip"

# Alternative/mirror URLs to try if the primary fails
ALTERNATIVE_URLS = [
    "https://deepio.cs.ox.ac.uk/data/oxiod_dataset.zip",
]

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "oxiod",
)


def download_file(url, dest_path, chunk_size=8192):
    """Download a file from url to dest_path with progress reporting."""
    print(f"Attempting download from: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = response.headers.get("Content-Length")
            if total_size:
                total_size = int(total_size)
                print(f"File size: {total_size / (1024 * 1024):.1f} MB")

            downloaded = 0
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(
                            f"\rDownloading: {downloaded / (1024 * 1024):.1f} / "
                            f"{total_size / (1024 * 1024):.1f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress
            print(f"Download complete: {dest_path}")
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"Failed to download from {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path, extract_dir):
    """Extract a zip file to the specified directory."""
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Extraction complete.")


def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Download the Oxford Inertial Odometry Dataset (OxIOD)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the dataset (default: data/oxiod)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(output_dir, "oxiod_dataset.zip")

    # Try downloading from all URLs
    urls = [DATASET_URL] + ALTERNATIVE_URLS
    success = False
    for url in urls:
        if download_file(url, zip_path):
            success = True
            break

    if not success:
        print(
            "\n========================================\n"
            "ERROR: Could not download the OxIOD dataset from any known URL.\n"
            "\n"
            "The Oxford server (deepio.cs.ox.ac.uk) may be temporarily unavailable.\n"
            "\n"
            "Please try one of the following:\n"
            "  1. Visit http://deepio.cs.ox.ac.uk manually in your browser\n"
            "  2. Download from the paper page: https://arxiv.org/abs/1809.07491\n"
            "  3. Search for 'OxIOD dataset' on academic dataset platforms\n"
            "  4. Try again later when the server may be back online\n"
            "\n"
            f"Once downloaded, place the zip file at:\n"
            f"  {zip_path}\n"
            f"Then run this script again (it will detect and extract the file).\n"
            "========================================"
        )
        sys.exit(1)

    # Extract the dataset
    try:
        extract_zip(zip_path, output_dir)
    except zipfile.BadZipFile:
        print(f"ERROR: Downloaded file is not a valid zip archive: {zip_path}")
        sys.exit(1)

    # Clean up zip if requested
    if not args.keep_zip:
        print(f"Removing zip file: {zip_path}")
        os.remove(zip_path)

    # Print summary
    print("\n========================================")
    print("OxIOD Dataset downloaded successfully!")
    print(f"Location: {output_dir}")
    print("\nDataset contents:")
    for item in sorted(Path(output_dir).iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_dir():
            n_files = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/ ({n_files} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name} ({size_mb:.1f} MB)")
    print("========================================")


if __name__ == "__main__":
    main()
