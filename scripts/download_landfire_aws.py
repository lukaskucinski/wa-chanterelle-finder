"""
Download LANDFIRE Existing Vegetation Type (EVT) data.

LANDFIRE now uses a dynamic download portal - direct URLs no longer work.
This script provides instructions and can download from alternative sources.
"""

import os
import sys
import zipfile
import webbrowser
from pathlib import Path
import requests
from tqdm import tqdm

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent
VEG_DIR = PROJECT_ROOT / "data" / "raw" / "vegetation"
VEG_DIR.mkdir(parents=True, exist_ok=True)

# Study area bounds for clipping request
BOUNDS = {
    "north": 49.0,
    "south": 45.5,
    "east": -120.5,
    "west": -122.5,
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"  Connecting...")
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()

        # Check if we got actual data or an error page
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print(f"  Received HTML instead of data (likely error page)")
            return False

        total_size = int(response.headers.get('content-length', 0))
        if total_size < 1000000:  # Less than 1MB is suspicious for this data
            print(f"  File too small ({total_size} bytes) - likely not the actual data")
            return False

        size_mb = total_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=f"  Downloading", leave=True) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def print_manual_instructions():
    """Print manual download instructions."""
    print("""
============================================================
MANUAL DOWNLOAD INSTRUCTIONS
============================================================

LANDFIRE now requires using their interactive portal. Follow these steps:

METHOD 1: LANDFIRE Viewer (Recommended - smaller download)
----------------------------------------------------------
1. Open: https://landfire.gov/viewer/

2. Navigate to Washington State:
   - Use the search box or zoom to the Cascade Range
   - Or enter coordinates: 47.5°N, -121.5°W

3. Click "Get Data" button (top right)

4. Draw a bounding box over the Cascades:
   - North: 49.0°
   - South: 45.5°
   - East: -120.5°
   - West: -122.5°

5. Select data product:
   - Version: "LF 2022" (or latest available)
   - Product: "Existing Vegetation Type (EVT)"

6. Click "Download" and wait for the job to process

7. Extract the downloaded zip to:
   """ + str(VEG_DIR) + """


METHOD 2: USGS EarthExplorer (Alternative)
----------------------------------------------------------
1. Open: https://earthexplorer.usgs.gov/

2. Set search criteria:
   - Click "Use Map" and draw a box over Washington Cascades
   - Or enter coordinates manually

3. Select datasets:
   - Data Sets → Vegetation Indices → LANDFIRE
   - Check "EVT - Existing Vegetation Type"

4. Download results


METHOD 3: Direct LFPS Portal
----------------------------------------------------------
1. Open: https://lfps.usgs.gov/helpdocs/productstable.html

2. Find "Existing Vegetation Type" in the table

3. Click the download link for LF 2022


After downloading, ensure the .tif file is in:
""" + str(VEG_DIR) + """

Then run: python scripts/03_preprocess_vegetation.py
""")


def open_landfire_viewer():
    """Open LANDFIRE viewer in browser."""
    url = "https://landfire.gov/viewer/"
    print(f"\nOpening LANDFIRE Viewer: {url}")
    try:
        webbrowser.open(url)
        return True
    except:
        print(f"Could not open browser. Please navigate to: {url}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("LANDFIRE EVT Data Download")
    print("=" * 60)
    print(f"\nOutput directory: {VEG_DIR}")

    # Check if already downloaded
    existing_tifs = list(VEG_DIR.glob("*EVT*.tif")) + list(VEG_DIR.glob("*evt*.tif"))
    if existing_tifs:
        print(f"\nEVT data already exists:")
        for f in existing_tifs:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print("\nSkipping download. Delete existing files to re-download.")
        return

    print("""
LANDFIRE has migrated to a new download system that requires
interactive selection through their web portal.

Automated download is no longer available for this dataset.
""")

    # Ask user what to do
    print("Options:")
    print("  [1] Open LANDFIRE Viewer in browser (recommended)")
    print("  [2] Show manual download instructions")
    print("  [3] Exit")

    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "3"

    if choice == "1":
        open_landfire_viewer()
        print("\n" + "-" * 60)
        print("After downloading from the LANDFIRE Viewer:")
        print("-" * 60)
        print(f"1. Extract the zip file")
        print(f"2. Copy the .tif file to: {VEG_DIR}")
        print(f"3. Run: python scripts/03_preprocess_vegetation.py")

    elif choice == "2":
        print_manual_instructions()

    else:
        print("\nExiting. Run this script again when ready to download.")


if __name__ == "__main__":
    main()
