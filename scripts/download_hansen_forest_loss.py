"""
Download Hansen Global Forest Change data for Washington State.

The Hansen/UMD Global Forest Change dataset (v1.12) provides:
- Annual forest loss detection from 2001-2024
- 30m resolution (matches our project)
- Derived from Landsat time-series analysis

Key bands:
- lossyear: Year of forest loss (1-24 = 2001-2024), 0 = no loss
- treecover2000: Tree canopy cover percentage in year 2000
- gain: Forest gain between 2000-2012 (binary)

Reference: Hansen et al., Science 2013
License: CC BY 4.0

Data tiles are 10x10 degrees. Washington State requires:
- 50N_130W (covers western WA: 40-50°N, 130-120°W)
- 50N_120W (covers eastern WA: 40-50°N, 120-110°W)
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Tuple

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "forest_loss"

# Hansen GFC 2024 v1.12 base URL
BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"

# Tiles covering Washington State (10x10 degree tiles)
# Tile names indicate the upper-left corner of each tile
TILES = [
    "50N_130W",  # Western WA (Olympics, west Cascades, coast)
    "50N_120W",  # Eastern WA (east Cascades, Columbia Basin)
]

# Bands to download
# - lossyear: Most important - year of forest loss (1-24 = 2001-2024)
# - treecover2000: Baseline canopy cover (useful for validation)
BANDS = [
    "lossyear",      # Year of gross forest cover loss (1-24 = 2001-2024)
    "treecover2000", # Tree canopy cover for year 2000 (0-100%)
]


def get_download_urls() -> List[Tuple[str, str, Path]]:
    """Generate list of (url, description, output_path) tuples."""
    downloads = []

    for tile in TILES:
        for band in BANDS:
            filename = f"Hansen_GFC-2024-v1.12_{band}_{tile}.tif"
            url = f"{BASE_URL}/{filename}"
            output_path = OUTPUT_DIR / filename
            description = f"{band} for tile {tile}"
            downloads.append((url, description, output_path))

    return downloads


def download_file(url: str, output_path: Path, description: str) -> bool:
    """Download a single file with progress indication."""
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return True

    print(f"  Downloading: {description}")
    print(f"    URL: {url}")

    try:
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r    Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, output_path, reporthook)
        print()  # New line after progress
        print(f"    Saved: {output_path.name}")
        return True

    except Exception as e:
        print(f"\n    ERROR: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        return False


def verify_downloads() -> None:
    """Verify downloaded files and print summary."""
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for tile in TILES:
        print(f"\nTile {tile}:")
        for band in BANDS:
            filename = f"Hansen_GFC-2024-v1.12_{band}_{tile}.tif"
            filepath = OUTPUT_DIR / filename

            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✓ {band}: {size_mb:.1f} MB")
            else:
                print(f"  ✗ {band}: NOT DOWNLOADED")


def print_next_steps() -> None:
    """Print instructions for processing the data."""
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. Process the Hansen data to create a forest loss year layer:
   python scripts/process_hansen_forest_loss.py

2. This will create: data/processed/forest_loss_year.tif
   - Values 0 = no loss detected
   - Values 1-24 = loss year (2001-2024)

3. Update forest age calculation to incorporate Hansen data:
   - Compare with LANDFIRE disturbance data
   - Use more recent of the two for each pixel

Data interpretation:
- lossyear: Pixel value indicates year of forest loss
  - 0 = No loss detected 2001-2024
  - 1 = Loss in 2001
  - 24 = Loss in 2024

- treecover2000: Baseline canopy cover percentage
  - Useful for identifying forested areas
  - Values 0-100 (percent tree cover)

Note: Hansen data captures ALL forest loss (logging, fire, disease,
development, etc.) but doesn't distinguish between causes.
""")


def main():
    """Main download workflow."""
    print("=" * 60)
    print("Hansen Global Forest Change Data Download")
    print("Version: GFC-2024-v1.12 (2000-2024)")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Get download list
    downloads = get_download_urls()
    print(f"\nFiles to download: {len(downloads)}")

    # Download each file
    print("\n" + "-" * 60)
    print("Downloading files...")
    print("-" * 60)

    success_count = 0
    for url, description, output_path in downloads:
        if download_file(url, output_path, description):
            success_count += 1

    # Verify and summarize
    verify_downloads()

    if success_count == len(downloads):
        print("\n✓ All files downloaded successfully!")
        print_next_steps()
    else:
        print(f"\n⚠ {len(downloads) - success_count} file(s) failed to download")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
