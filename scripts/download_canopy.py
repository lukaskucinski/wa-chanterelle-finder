"""
Download NLCD Tree Canopy Cover data.

NLCD and USFS tree canopy data requires interactive download through
their web portals. This script provides instructions and opens the
appropriate download page.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent
CANOPY_DIR = PROJECT_ROOT / "data" / "raw" / "canopy"
CANOPY_DIR.mkdir(parents=True, exist_ok=True)


def print_manual_instructions():
    """Print manual download instructions."""
    print("""
============================================================
NLCD TREE CANOPY COVER DOWNLOAD INSTRUCTIONS
============================================================

The NLCD Tree Canopy Cover data requires interactive download.
Choose one of these methods:

METHOD 1: USFS Raster Gateway (Recommended - smaller regions)
-------------------------------------------------------------
1. Open: https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/

2. Under "CONUS" section, find "NLCD TCC" dropdown

3. Select year: 2021 (or latest available)

4. Click the download button

5. Save to:
   """ + str(CANOPY_DIR) + """


METHOD 2: MRLC Viewer (Region-specific download)
-------------------------------------------------------------
1. Open: https://www.mrlc.gov/viewer/

2. Zoom to Washington State / Cascade Range

3. Click "Download" tool

4. Draw a box over the study area:
   - North: 49.0°
   - South: 45.5°
   - East: -120.5°
   - West: -122.5°

5. Select "Tree Canopy Cover" product

6. Download and extract


METHOD 3: USGS ScienceBase
-------------------------------------------------------------
1. Open: https://www.sciencebase.gov/catalog/item/649595e9d34ef77fcb01dca3

2. Look for download links or external links

3. Navigate to MRLC or USFS download pages


EXPECTED FILE:
- nlcd_tcc_conus_2021_v2021-4.tif (or .zip)
- Size: ~3-4 GB for full CONUS, ~200-500 MB for regional clip

After downloading, ensure the .tif file is in:
""" + str(CANOPY_DIR) + """

Then run: python scripts/03_preprocess_vegetation.py
""")


def open_download_page():
    """Open the USFS download page in browser."""
    url = "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/"
    print(f"\nOpening USFS Tree Canopy Download page: {url}")
    try:
        webbrowser.open(url)
        return True
    except:
        print(f"Could not open browser. Please navigate to: {url}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("NLCD Tree Canopy Cover Download")
    print("=" * 60)
    print(f"\nOutput directory: {CANOPY_DIR}")

    # Check if already downloaded
    existing_tifs = list(CANOPY_DIR.glob("*tcc*.tif")) + list(CANOPY_DIR.glob("*canopy*.tif"))
    if existing_tifs:
        print(f"\nTree canopy data already exists:")
        for f in existing_tifs:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print("\nSkipping download. Delete existing files to re-download.")
        return

    print("""
NLCD Tree Canopy Cover data requires interactive download
through the USFS or MRLC web portals.
""")

    # Ask user what to do
    print("Options:")
    print("  [1] Open USFS download page in browser (recommended)")
    print("  [2] Open MRLC Viewer for regional download")
    print("  [3] Show detailed instructions")
    print("  [4] Exit")

    try:
        choice = input("\nEnter choice (1/2/3/4): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "4"

    if choice == "1":
        open_download_page()
        print("\n" + "-" * 60)
        print("On the USFS page:")
        print("-" * 60)
        print("1. Find 'CONUS' section → 'NLCD TCC' dropdown")
        print("2. Select year 2021")
        print("3. Click download button")
        print(f"4. Save/extract to: {CANOPY_DIR}")
        print("5. Run: python scripts/03_preprocess_vegetation.py")

    elif choice == "2":
        url = "https://www.mrlc.gov/viewer/"
        print(f"\nOpening MRLC Viewer: {url}")
        try:
            webbrowser.open(url)
        except:
            print(f"Could not open browser. Navigate to: {url}")
        print("\nUse the Download tool to select Washington Cascades region.")

    elif choice == "3":
        print_manual_instructions()

    else:
        print("\nExiting. Run this script again when ready to download.")


if __name__ == "__main__":
    main()
