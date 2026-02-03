"""
Download LANDFIRE Existing Vegetation Type (EVT) data.

LANDFIRE data is hosted on AWS S3 as part of the USFS data distribution.
This script downloads EVT data for the Pacific Northwest region.

Note: CONUS-wide data is very large (~2-3 GB). This script attempts
to download the full dataset which you can then clip to the study area.
"""

import os
import sys
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent
VEG_DIR = PROJECT_ROOT / "data" / "raw" / "vegetation"
VEG_DIR.mkdir(parents=True, exist_ok=True)

# LANDFIRE data URLs
# LF 2022 (also known as LF2022/LF 2.3.0) EVT data
LANDFIRE_URLS = {
    # Primary: AWS hosted LANDFIRE data
    "aws_lf2022": "https://s3-us-west-2.amazonaws.com/landfire/LF2022/LF2022_EVT_220_CONUS/LF2022_EVT_220_CONUS.zip",
    # Alternative: LANDFIRE direct (may require session)
    "landfire_direct": "https://landfire.gov/bulk/downloadfile.php?TYPE=landfire&FNAME=LF2022_EVT_220_CONUS.zip",
    # Older version as fallback
    "aws_lf2020": "https://s3-us-west-2.amazonaws.com/landfire/LF2020/LF2020_EVT_200_CONUS/LF2020_EVT_200_CONUS.zip",
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"  Connecting to: {url[:80]}...")
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)

        if response.status_code == 404:
            print(f"  404 Not Found")
            return False

        if response.status_code == 403:
            print(f"  403 Forbidden - may require authentication")
            return False

        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        if total_size == 0:
            print("  Warning: Unknown file size")

        size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        print(f"  File size: {size_mb:.1f} MB")

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=f"  Downloading", leave=True) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a zip file."""
    try:
        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List contents
            file_list = zf.namelist()
            print(f"  Archive contains {len(file_list)} files")

            # Extract
            zf.extractall(extract_dir)

        print(f"  Extracted to: {extract_dir}")
        return True

    except zipfile.BadZipFile:
        print(f"  Error: Bad zip file")
        return False
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def main():
    """Download LANDFIRE EVT data."""
    print("=" * 60)
    print("LANDFIRE EVT Data Download")
    print("=" * 60)
    print(f"\nOutput directory: {VEG_DIR}")

    # Check if already downloaded
    existing_tifs = list(VEG_DIR.glob("*EVT*.tif"))
    if existing_tifs:
        print(f"\nEVT data already exists:")
        for f in existing_tifs:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print("\nSkipping download. Delete existing files to re-download.")
        return

    print("\n" + "-" * 60)
    print("Attempting download from available sources...")
    print("-" * 60)
    print("\nNote: LANDFIRE CONUS data is large (2-3 GB).")
    print("This may take 10-30 minutes depending on connection.\n")

    downloaded = False
    zip_path = None

    for source_name, url in LANDFIRE_URLS.items():
        print(f"\n[{source_name.upper()}]")

        # Determine filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename.endswith(".zip"):
            filename = f"LF_EVT_{source_name}.zip"

        zip_path = VEG_DIR / filename

        if download_file(url, zip_path):
            downloaded = True
            print(f"  Success!")
            break
        else:
            # Clean up partial download
            if zip_path.exists():
                zip_path.unlink()
            print(f"  Failed, trying next source...")

    if not downloaded:
        print("\n" + "=" * 60)
        print("AUTOMATED DOWNLOAD FAILED")
        print("=" * 60)
        print("""
All automated download attempts failed. Please download manually:

1. Go to: https://landfire.gov/viewer/
2. Click "Get Data" → "LF 2022"
3. Select "Existing Vegetation Type (EVT)"
4. For Area: Select "CONUS" or draw a box over Washington State
5. Download and extract to:
   """ + str(VEG_DIR) + """

Alternative: Use USGS EarthExplorer
1. Go to: https://earthexplorer.usgs.gov/
2. Set coordinates for Washington (45.5-49°N, 120.5-122.5°W)
3. Data Sets → Vegetation → LANDFIRE → EVT
4. Download and extract to the vegetation folder
""")
        return

    # Extract the zip file
    print("\n" + "-" * 60)
    print("Extracting data...")
    print("-" * 60)

    if zip_path and zip_path.exists():
        if extract_zip(zip_path, VEG_DIR):
            # Find the extracted TIF
            tif_files = list(VEG_DIR.glob("**/*EVT*.tif"))
            if tif_files:
                print(f"\nExtracted TIF files:")
                for f in tif_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  - {f.name} ({size_mb:.1f} MB)")

                    # Move to vegetation dir if in subdirectory
                    if f.parent != VEG_DIR:
                        dest = VEG_DIR / f.name
                        f.rename(dest)
                        print(f"    Moved to: {dest}")

            # Optionally remove zip to save space
            print(f"\nKeeping zip file: {zip_path.name}")
            print("Delete manually if you want to save disk space.")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    final_tifs = list(VEG_DIR.glob("*EVT*.tif"))
    if final_tifs:
        for f in final_tifs:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
        print("\nNext step: Run python scripts/03_preprocess_vegetation.py")
    else:
        print("  No TIF files found. Check extraction manually.")


if __name__ == "__main__":
    main()
