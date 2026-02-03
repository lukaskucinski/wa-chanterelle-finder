"""
Download USGS 3DEP DEM tiles from AWS S3.

USGS 3DEP data is hosted on AWS Open Data Registry.
No authentication required - public bucket.

Tiles needed for Washington Cascades study area:
- Bounds: -122.5 to -120.5 (lon), 45.5 to 49.0 (lat)
- Resolution: 1/3 arc-second (~10m)
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent
DEM_DIR = PROJECT_ROOT / "data" / "raw" / "dem"
DEM_DIR.mkdir(parents=True, exist_ok=True)

# AWS S3 base URL for USGS 3DEP 1/3 arc-second DEMs
BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"

# Study area bounds
SOUTH_LAT = 45.5
NORTH_LAT = 49.0
WEST_LON = -122.5
EAST_LON = -120.5


def get_tile_name(lat: int, lon: int) -> str:
    """
    Generate tile name for given lat/lon.

    Tile naming: USGS_13_n##w###.tif
    - n## = latitude of southern edge
    - w### = absolute longitude of western edge
    """
    lat_str = f"n{lat:02d}"
    lon_str = f"w{abs(lon):03d}"
    return f"USGS_13_{lat_str}{lon_str}.tif"


def get_required_tiles() -> list:
    """
    Calculate which tiles are needed for the study area.

    Each tile covers 1x1 degree.
    """
    tiles = []

    # Latitude range (southern edge of each tile)
    lat_min = int(SOUTH_LAT)  # 45
    lat_max = int(NORTH_LAT)  # 49, but tile n48 covers up to 49

    # Longitude range (western edge of each tile, absolute values)
    lon_min = int(abs(WEST_LON))   # 122 (covers 122-123°W, but we need 123 too)
    lon_max = int(abs(EAST_LON))   # 120 (covers 120-121°W)

    # Need to include the tile that extends beyond our bounds
    for lat in range(lat_min, lat_max + 1):  # 45, 46, 47, 48
        for lon in range(lon_max, lon_min + 2):  # 120, 121, 122, 123
            tile_name = get_tile_name(lat, lon)
            tiles.append(tile_name)

    return tiles


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code == 404:
            print(f"  Not found (may not exist for this area)")
            return False

        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=f"  {dest_path.name}", leave=False) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
        return False


def main():
    """Download all required DEM tiles."""
    print("=" * 60)
    print("USGS 3DEP DEM Download from AWS S3")
    print("=" * 60)
    print(f"\nStudy area:")
    print(f"  Latitude:  {SOUTH_LAT}° to {NORTH_LAT}°N")
    print(f"  Longitude: {WEST_LON}° to {EAST_LON}°E")
    print(f"\nOutput directory: {DEM_DIR}")

    tiles = get_required_tiles()
    print(f"\nTiles to download: {len(tiles)}")
    for tile in tiles:
        print(f"  - {tile}")

    print("\n" + "-" * 60)
    print("Downloading tiles...")
    print("-" * 60)

    downloaded = 0
    skipped = 0
    failed = 0

    for tile in tiles:
        dest_path = DEM_DIR / tile

        # Skip if already exists
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"\n[SKIP] {tile} (already exists, {size_mb:.1f} MB)")
            skipped += 1
            continue

        # Construct URL
        # URL pattern: BASE_URL/n##w###/USGS_13_n##w###.tif
        tile_dir = tile.replace("USGS_13_", "").replace(".tif", "")
        url = f"{BASE_URL}/{tile_dir}/{tile}"

        print(f"\n[DOWNLOAD] {tile}")
        print(f"  URL: {url}")

        if download_file(url, dest_path):
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"  Success: {size_mb:.1f} MB")
            downloaded += 1
        else:
            failed += 1
            # Clean up partial download
            if dest_path.exists():
                dest_path.unlink()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed/Not found: {failed}")

    # List downloaded files
    dem_files = list(DEM_DIR.glob("*.tif"))
    if dem_files:
        total_size = sum(f.stat().st_size for f in dem_files) / (1024 * 1024)
        print(f"\nTotal DEM files: {len(dem_files)} ({total_size:.1f} MB)")

    print("\nNext step: Run python scripts/02_preprocess_dem.py")


if __name__ == "__main__":
    main()
