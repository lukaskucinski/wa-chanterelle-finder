"""
Download POLARIS soil pH data for Washington State.

POLARIS provides 30m soil properties derived from SSURGO.
We download the 0-5cm depth (surface pH) which is most relevant
for chanterelle mycorrhizal associations.

Source: http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/
"""

import os
import urllib.request
from pathlib import Path
from itertools import product

# Configuration
BASE_URL = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/ph/mean/0_5"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "soil" / "polaris_ph"

# Washington State bounding box (with buffer)
# Lat: ~45.5 to ~49.0, Lon: ~-124.8 to ~-116.9
# Tile naming: lat4546 = 45-46°N, lon-125-124 = 125°W to 124°W

# Latitude tiles needed (south to north)
LAT_TILES = ["4546", "4647", "4748", "4849"]

# Longitude tiles needed (west to east)
# Note: POLARIS uses format lon-125-124 meaning 125°W to 124°W
LON_TILES = [
    "-125-124", "-124-123", "-123-122", "-122-121",
    "-121-120", "-120-119", "-119-118", "-118-117"
]


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output path."""
    try:
        print(f"  Downloading: {output_path.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("Not found (no data for this tile)")
        else:
            print(f"HTTP Error {e.code}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Download POLARIS pH tiles for Washington State."""
    print("=" * 60)
    print("POLARIS Soil pH Data Download")
    print("=" * 60)
    print(f"\nSource: {BASE_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Depth: 0-5 cm (surface)")
    print(f"\nTiles to download: {len(LAT_TILES)} lat x {len(LON_TILES)} lon = {len(LAT_TILES) * len(LON_TILES)} max")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download tiles
    downloaded = 0
    skipped = 0
    not_found = 0

    print("\nDownloading tiles...")
    for lat, lon in product(LAT_TILES, LON_TILES):
        filename = f"lat{lat}_lon{lon}.tif"
        url = f"{BASE_URL}/{filename}"
        output_path = OUTPUT_DIR / filename

        # Skip if already exists
        if output_path.exists():
            print(f"  Skipping (exists): {filename}")
            skipped += 1
            continue

        if download_file(url, output_path):
            downloaded += 1
        else:
            not_found += 1

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Not found: {not_found}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")

    # List downloaded files
    tif_files = list(OUTPUT_DIR.glob("*.tif"))
    if tif_files:
        total_size = sum(f.stat().st_size for f in tif_files) / (1024 * 1024)
        print(f"Total files: {len(tif_files)} ({total_size:.1f} MB)")


if __name__ == "__main__":
    main()
