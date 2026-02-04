"""
Script 02: Preprocess Digital Elevation Model data.

This script:
1. Merges DEM tiles if multiple exist
2. Clips to study area
3. Reprojects to UTM 10N (EPSG:32610)
4. Resamples to 30m resolution
5. Calculates slope and aspect derivatives
6. Converts elevation to feet for scoring
"""

import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.raster_utils import (
    TARGET_CRS,
    TARGET_RESOLUTION,
    NODATA,
    read_raster,
    write_raster,
    reproject_raster,
    clip_raster_to_bounds,
    calculate_slope,
    calculate_aspect,
    meters_to_feet,
    get_raster_stats,
    get_study_area_bounds,
)

# Directories
RAW_DEM_DIR = PROJECT_ROOT / "data" / "raw" / "dem"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def merge_dem_tiles(input_dir: Path, output_path: Path) -> Path:
    """
    Merge multiple DEM tiles into a single raster.

    Args:
        input_dir: Directory containing DEM tiles
        output_path: Output merged raster path

    Returns:
        Path to merged raster
    """
    # Find all DEM files
    dem_patterns = ["*.tif", "*.tiff", "*.img"]
    dem_files = []
    for pattern in dem_patterns:
        dem_files.extend(input_dir.glob(pattern))

    if not dem_files:
        raise FileNotFoundError(f"No DEM files found in {input_dir}")

    print(f"Found {len(dem_files)} DEM tile(s)")

    if len(dem_files) == 1:
        # Single file - just copy/reference it
        print(f"Single tile: {dem_files[0]}")
        return dem_files[0]

    # Open all tiles
    src_files = [rasterio.open(f) for f in dem_files]

    try:
        # Merge tiles
        print("Merging DEM tiles...")
        mosaic, out_transform = merge(src_files)

        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "BIGTIFF": "YES",
        })

        # Write merged raster
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        print(f"Merged DEM saved to: {output_path}")
        return output_path

    finally:
        # Close all source files
        for src in src_files:
            src.close()


def process_dem():
    """Main DEM processing workflow."""
    print("=" * 60)
    print("DEM PREPROCESSING")
    print("=" * 60)

    # Output paths
    merged_path = PROCESSED_DIR / "dem_merged.tif"
    clipped_path = PROCESSED_DIR / "dem_clipped.tif"
    reprojected_path = PROCESSED_DIR / "dem_utm.tif"
    elevation_ft_path = PROCESSED_DIR / "elevation_ft.tif"
    slope_path = PROCESSED_DIR / "slope_degrees.tif"
    aspect_path = PROCESSED_DIR / "aspect_degrees.tif"

    # Step 1: Merge tiles (if needed)
    if not merged_path.exists():
        print("\n[1/5] Merging DEM tiles...")
        if RAW_DEM_DIR.exists() and any(RAW_DEM_DIR.glob("*.tif")):
            merge_dem_tiles(RAW_DEM_DIR, merged_path)
        else:
            print(f"ERROR: No DEM files found in {RAW_DEM_DIR}")
            print("Please download DEM data first. See 01_download_data.py")
            return
    else:
        print(f"\n[1/5] Using existing merged DEM: {merged_path}")

    # Step 2: Clip to study area
    if not clipped_path.exists():
        print("\n[2/5] Clipping to study area...")
        clip_raster_to_bounds(merged_path, clipped_path)
        print(f"Clipped DEM saved to: {clipped_path}")
    else:
        print(f"\n[2/5] Using existing clipped DEM: {clipped_path}")

    # Step 3: Reproject to UTM and resample to 30m
    if not reprojected_path.exists():
        print("\n[3/5] Reprojecting to UTM 10N (30m resolution)...")
        reproject_raster(
            clipped_path,
            reprojected_path,
            dst_crs=TARGET_CRS,
            dst_resolution=TARGET_RESOLUTION,
            resampling=Resampling.bilinear
        )
        print(f"Reprojected DEM saved to: {reprojected_path}")
    else:
        print(f"\n[3/5] Using existing reprojected DEM: {reprojected_path}")

    # Read the processed DEM
    with rasterio.open(reprojected_path) as src:
        dem_m = src.read(1)
        transform = src.transform
        nodata = src.nodata or NODATA

    # Handle nodata
    dem_m = np.where(dem_m == nodata, np.nan, dem_m)

    # Step 4: Convert to feet and calculate derivatives
    if not elevation_ft_path.exists():
        print("\n[4/5] Converting elevation to feet...")
        dem_ft = meters_to_feet(dem_m)
        dem_ft = np.where(np.isnan(dem_ft), NODATA, dem_ft)
        write_raster(elevation_ft_path, dem_ft, transform, TARGET_CRS)
        print(f"Elevation (feet) saved to: {elevation_ft_path}")
    else:
        print(f"\n[4/5] Using existing elevation (feet): {elevation_ft_path}")

    # Step 5: Calculate slope and aspect
    if not slope_path.exists() or not aspect_path.exists():
        print("\n[5/5] Calculating slope and aspect...")

        if not slope_path.exists():
            slope = calculate_slope(dem_m, TARGET_RESOLUTION)
            slope = np.where(np.isnan(slope), NODATA, slope)
            write_raster(slope_path, slope, transform, TARGET_CRS)
            print(f"Slope saved to: {slope_path}")

        if not aspect_path.exists():
            aspect = calculate_aspect(dem_m, TARGET_RESOLUTION)
            aspect = np.where(np.isnan(aspect), NODATA, aspect)
            write_raster(aspect_path, aspect, transform, TARGET_CRS)
            print(f"Aspect saved to: {aspect_path}")
    else:
        print(f"\n[5/5] Using existing slope and aspect")

    # Print statistics
    print("\n" + "-" * 40)
    print("DEM STATISTICS")
    print("-" * 40)

    for name, path in [
        ("Elevation (ft)", elevation_ft_path),
        ("Slope (degrees)", slope_path),
        ("Aspect (degrees)", aspect_path),
    ]:
        if path.exists():
            stats = get_raster_stats(path)
            print(f"\n{name}:")
            print(f"  Min: {stats['min']:.1f}")
            print(f"  Max: {stats['max']:.1f}")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  Valid cells: {stats['valid_count']:,}")

    print("\n" + "=" * 60)
    print("DEM preprocessing complete!")
    print("=" * 60)


def main():
    """Entry point."""
    process_dem()


if __name__ == "__main__":
    main()
