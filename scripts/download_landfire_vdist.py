"""
Download LANDFIRE Vegetation Disturbance data for forest age estimation.

LANDFIRE provides Historical Disturbance (HDist) data which contains the year
of the last disturbance event (fire, harvest, etc.). We can calculate
Time Since Disturbance (TSD) = Current Year - Disturbance Year.

For chanterelles, second-growth forests (40-80 years old) are optimal.

LANDFIRE Data Access Options:
1. LANDFIRE Map Viewer (manual): https://www.landfire.gov/viewer/
2. Bulk download: https://landfire.gov/version_download.php
3. Direct file access (used here): S3-hosted COGs

This script downloads HDist data tiles covering Washington State.
"""

import os
import urllib.request
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "vegetation" / "landfire_hdist"

# LANDFIRE 2022 Historical Disturbance (HDist) - S3 hosted
# HDist contains year of last disturbance (e.g., 2015, 2020, etc.)
# Values: Year of disturbance, or 0/NoData for no recorded disturbance

# LANDFIRE provides CONUS-wide data, but it's very large (~4GB)
# For Washington, we can use the LANDFIRE Data Distribution Site
# or download regional tiles

# Direct LANDFIRE HTTPS endpoints for LF2022 products
LANDFIRE_BASE = "https://s3-us-west-2.amazonaws.com/landfire/LF2022/LF2022_HDist_220_CONUS/LF2022_HDist_220_CONUS"

# Alternative: Use LANDFIRE's clipping service via their API
# This requires registration but provides state-level extracts

def print_manual_instructions():
    """Print instructions for manual download."""
    print("""
============================================================
LANDFIRE VDIST/HDist Manual Download Instructions
============================================================

The LANDFIRE download system requires either:
1. Manual download through their viewer, OR
2. Using their bulk download with CONUS-wide files

OPTION 1: LANDFIRE Viewer (Recommended for state-level data)
------------------------------------------------------------
1. Go to: https://www.landfire.gov/viewer/
2. Zoom to Washington State
3. Click "Get Data" in the toolbar
4. Select "LF 2022" version
5. Check "Disturbance" products:
   - HDist (Historical Disturbance Year)
   - Or VDist (Vegetation Disturbance)
6. Draw a rectangle around your study area
7. Download and extract to: data/raw/vegetation/landfire_hdist/

OPTION 2: Bulk Download (Large file - ~4GB for CONUS)
------------------------------------------------------------
1. Go to: https://landfire.gov/version_download.php
2. Select "LF 2022"
3. Find "LF2022_HDist_220_CONUS" (Historical Disturbance)
4. Download the GeoTIFF version
5. Extract to: data/raw/vegetation/landfire_hdist/

OPTION 3: LFPS (LANDFIRE Product Service) - API Access
------------------------------------------------------------
Requires registration at: https://lfps.usgs.gov/
Allows programmatic state-level downloads.

After downloading, run the processing script:
    python scripts/process_landfire_hdist.py
============================================================
""")


def check_existing_data():
    """Check if LANDFIRE data already exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Look for HDist files
    hdist_files = list(OUTPUT_DIR.glob("*HDist*.tif")) + list(OUTPUT_DIR.glob("*hdist*.tif"))
    hdist_files += list(OUTPUT_DIR.glob("*HDIST*.tif"))

    # Also check for VDIST files
    vdist_files = list(OUTPUT_DIR.glob("*VDist*.tif")) + list(OUTPUT_DIR.glob("*vdist*.tif"))

    if hdist_files:
        print(f"Found existing HDist data: {hdist_files[0].name}")
        return hdist_files[0]
    elif vdist_files:
        print(f"Found existing VDist data: {vdist_files[0].name}")
        return vdist_files[0]

    return None


def try_direct_download():
    """
    Attempt direct download from LANDFIRE S3.
    Note: This downloads CONUS-wide data which is very large.
    """
    print("\nAttempting direct download from LANDFIRE S3...")
    print("WARNING: CONUS file is ~4GB. This may take a while.")

    # LANDFIRE doesn't allow direct partial downloads easily
    # The CONUS file is the main option for programmatic access
    url = "https://landfire.gov/bulk/downloadfile.php?FNAME=LF2022_HDist_220_CONUS.zip&TYPE=landfire"

    output_path = OUTPUT_DIR / "LF2022_HDist_220_CONUS.zip"

    try:
        print(f"Downloading from: {url}")
        print(f"This is a large file (~4GB). Consider using LANDFIRE Viewer instead.")

        # Only proceed if user confirms
        response = input("Continue with CONUS download? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return None

        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Direct download failed: {e}")
        return None


def main():
    """Main entry point."""
    print("=" * 60)
    print("LANDFIRE Historical Disturbance Data")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Check for existing data
    existing = check_existing_data()
    if existing:
        print(f"\nExisting data found! Ready for processing.")
        print(f"Run: python scripts/03_preprocess_vegetation.py")
        return

    # Print manual instructions
    print_manual_instructions()

    # Offer direct download option
    print("\nWould you like to attempt automatic download?")
    print("Note: This downloads the full CONUS file (~4GB)")

    try:
        response = input("Attempt automatic download? (y/n): ")
        if response.lower() == 'y':
            try_direct_download()
    except EOFError:
        # Non-interactive mode
        print("\nRunning in non-interactive mode.")
        print("Please download manually using the instructions above.")


if __name__ == "__main__":
    main()
