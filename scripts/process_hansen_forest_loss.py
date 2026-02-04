"""
Process Hansen Global Forest Change data for Washington Cascades.

This script:
1. Merges the two 10x10 degree tiles covering Washington
2. Clips to the study area
3. Reprojects to match our template (UTM 10N, 30m)
4. Creates a forest loss year raster for integration with forest age

The output can be combined with LANDFIRE disturbance data to create
a more complete picture of forest disturbance history.
"""

import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.crs import CRS
from shapely.geometry import box
import geopandas as gpd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.raster_utils import (
    TARGET_CRS,
    TARGET_RESOLUTION,
    NODATA,
    read_raster,
    write_raster,
    get_study_area_bounds,
    create_study_area_polygon,
)

# Directories
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "forest_loss"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Current year for age calculation
CURRENT_YEAR = 2024


def merge_tiles(band: str) -> Path:
    """Merge Hansen tiles for a specific band."""
    print(f"\n  Merging {band} tiles...")

    # Find all tiles for this band
    tile_files = list(RAW_DIR.glob(f"Hansen_GFC-2024-v1.12_{band}_*.tif"))

    if not tile_files:
        print(f"    ERROR: No tiles found for {band}")
        return None

    print(f"    Found {len(tile_files)} tiles")

    # Open all tiles
    src_files = [rasterio.open(f) for f in tile_files]

    # Merge
    mosaic, out_transform = merge(src_files)

    # Get metadata from first file
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw",
    })

    # Close source files
    for src in src_files:
        src.close()

    # Save merged file
    merged_path = RAW_DIR / f"hansen_{band}_merged.tif"
    with rasterio.open(merged_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    print(f"    Saved: {merged_path.name}")
    return merged_path


def clip_and_reproject(input_path: Path, template_path: Path, output_path: Path) -> bool:
    """Clip to study area and reproject to match template."""
    print(f"\n  Clipping and reprojecting...")

    # Get study area bounds
    bounds = get_study_area_bounds()

    # Create study area polygon in WGS84
    study_poly = create_study_area_polygon(bounds)

    with rasterio.open(input_path) as src:
        # Buffer in projected CRS then convert back
        study_projected = study_poly.to_crs(TARGET_CRS)
        buffered_projected = study_projected.buffer(50000)  # 50km buffer in meters
        buffered = buffered_projected.to_crs(study_poly.crs)

        try:
            clipped, clipped_transform = mask(
                src,
                buffered.geometry,
                crop=True,
                nodata=0
            )
        except Exception as e:
            print(f"    Clip error: {e}")
            return False

        # Get template info for reprojection
        with rasterio.open(template_path) as template:
            dst_crs = template.crs
            dst_transform = template.transform
            dst_width = template.width
            dst_height = template.height

        # Reproject to match template
        dst_data = np.zeros((1, dst_height, dst_width), dtype=src.dtypes[0])

        reproject(
            source=clipped,
            destination=dst_data,
            src_transform=clipped_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest  # Categorical data
        )

        # Save output
        profile = {
            'driver': 'GTiff',
            'dtype': dst_data.dtype,
            'width': dst_width,
            'height': dst_height,
            'count': 1,
            'crs': dst_crs,
            'transform': dst_transform,
            'nodata': 0,
            'compress': 'lzw',
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_data)

    print(f"    Saved: {output_path.name}")
    return True


def create_forest_loss_year(template_path: Path) -> Path:
    """
    Create aligned forest loss year raster from Hansen data.

    Output values:
    - 0: No forest loss detected (2001-2024)
    - 2001-2024: Year of forest loss
    """
    output_path = PROCESSED_DIR / "hansen_loss_year.tif"

    if output_path.exists():
        print(f"\n  Output already exists: {output_path}")
        return output_path

    # First, merge lossyear tiles
    merged_path = RAW_DIR / "hansen_lossyear_merged.tif"
    if not merged_path.exists():
        merged_path = merge_tiles("lossyear")
        if merged_path is None:
            return None

    # Clip and reproject to match template
    aligned_path = PROCESSED_DIR / "hansen_lossyear_aligned.tif"
    if not aligned_path.exists():
        if not clip_and_reproject(merged_path, template_path, aligned_path):
            return None

    # Convert from 1-24 encoding to actual years (2001-2024)
    print("\n  Converting to actual years...")

    data, meta = read_raster(aligned_path)

    # Hansen encodes years as 1-24 (meaning 2001-2024)
    # Convert to actual year values
    loss_year = np.zeros_like(data, dtype=np.int16)
    has_loss = (data > 0) & (data <= 24)
    loss_year[has_loss] = 2000 + data[has_loss].astype(np.int16)

    # Save with actual years
    write_raster(
        output_path,
        loss_year,
        meta['transform'],
        TARGET_CRS,
        nodata=0,
        dtype='int16'
    )

    print(f"    Saved: {output_path.name}")

    # Statistics
    valid = loss_year > 0
    if np.any(valid):
        print(f"\n  Hansen Forest Loss Statistics:")
        print(f"    Pixels with loss: {np.sum(valid):,}")
        print(f"    First loss year: {loss_year[valid].min()}")
        print(f"    Latest loss year: {loss_year[valid].max()}")

        # Yearly breakdown
        print(f"\n  Loss by decade:")
        for decade_start in [2001, 2011, 2021]:
            decade_end = min(decade_start + 9, 2024)
            decade_mask = (loss_year >= decade_start) & (loss_year <= decade_end)
            count = np.sum(decade_mask)
            print(f"    {decade_start}-{decade_end}: {count:,} pixels")

    return output_path


def combine_with_landfire(hansen_path: Path, template_path: Path) -> Path:
    """
    Combine Hansen forest loss with LANDFIRE disturbance data.

    Uses the MORE RECENT disturbance year from either dataset.
    """
    output_path = PROCESSED_DIR / "combined_disturbance_year.tif"

    # Check for LANDFIRE data - prefer aligned version
    landfire_path = PROCESSED_DIR / "landfire_disturbance_year.tif"

    if not landfire_path.exists():
        # Fall back to raw version (not aligned - will cause shape mismatch)
        landfire_path = PROJECT_ROOT / "data" / "raw" / "vegetation" / "landfire_hdist" / "last_disturbance_year_WA.tif"

    if not landfire_path.exists():
        print("\n  LANDFIRE disturbance data not found")
        print(f"    Expected: {PROCESSED_DIR / 'landfire_disturbance_year.tif'}")
        print("    Run: python scripts/process_annual_disturbance.py")
        print("    Skipping combination - using Hansen only")
        return hansen_path

    print("\n  Combining Hansen with LANDFIRE disturbance data...")
    print(f"    Hansen: {hansen_path.name}")
    print(f"    LANDFIRE: {landfire_path.name}")

    # Read both datasets
    hansen_data, hansen_meta = read_raster(hansen_path)
    landfire_data, _ = read_raster(landfire_path)

    print(f"    Hansen shape: {hansen_data.shape}")
    print(f"    LANDFIRE shape: {landfire_data.shape}")

    # Ensure same shape
    if hansen_data.shape != landfire_data.shape:
        print(f"    ERROR: Shape mismatch!")
        print("    To fix: re-run python scripts/process_annual_disturbance.py")
        print("    This will create an aligned LANDFIRE file.")
        print("    Skipping combination - using Hansen only")
        return hansen_path

    print("    Shapes match! Combining datasets...")

    # Take the more recent disturbance year from either source
    combined = np.maximum(hansen_data, landfire_data)

    write_raster(
        output_path,
        combined.astype(np.int16),
        hansen_meta['transform'],
        TARGET_CRS,
        nodata=0,
        dtype='int16'
    )

    print(f"    Saved: {output_path.name}")

    # Compare statistics
    hansen_valid = hansen_data > 0
    landfire_valid = landfire_data > 0
    combined_valid = combined > 0

    print(f"\n  Disturbance Coverage Comparison:")
    print(f"    Hansen only:   {np.sum(hansen_valid):,} pixels")
    print(f"    LANDFIRE only: {np.sum(landfire_valid):,} pixels")
    print(f"    Combined:      {np.sum(combined_valid):,} pixels")

    # Where do they differ?
    both = hansen_valid & landfire_valid
    hansen_newer = both & (hansen_data > landfire_data)
    landfire_newer = both & (landfire_data > hansen_data)

    print(f"\n  Where both detected disturbance ({np.sum(both):,} pixels):")
    print(f"    Hansen more recent:   {np.sum(hansen_newer):,} pixels")
    print(f"    LANDFIRE more recent: {np.sum(landfire_newer):,} pixels")

    return output_path


def main():
    """Main processing workflow."""
    print("=" * 60)
    print("Hansen Global Forest Change Processing")
    print("=" * 60)

    # Check for input data
    tile_files = list(RAW_DIR.glob("Hansen_*.tif"))
    if not tile_files:
        print(f"\nERROR: No Hansen data found in {RAW_DIR}")
        print("Please run: python scripts/download_hansen_forest_loss.py")
        return

    print(f"\nFound {len(tile_files)} Hansen tiles")

    # Check for template
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    if not template_path.exists():
        print(f"\nERROR: Template not found: {template_path}")
        print("Please run: python scripts/02_preprocess_dem.py")
        return

    # Process Hansen data
    print("\n[1/3] Processing Hansen forest loss year...")
    hansen_path = create_forest_loss_year(template_path)

    if hansen_path is None:
        print("\nERROR: Failed to process Hansen data")
        return

    # Combine with LANDFIRE (if available)
    print("\n[2/3] Combining with LANDFIRE disturbance data...")
    combined_path = combine_with_landfire(hansen_path, template_path)

    # Update forest age using combined data
    print("\n[3/3] Summary...")
    print(f"\n  Outputs:")
    print(f"    Hansen loss year:     {hansen_path}")
    print(f"    Combined disturbance: {combined_path}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print("""
Next steps:
1. Delete existing forest age: rm data/processed/forest_age.tif
2. Rerun vegetation preprocessing: python scripts/03_preprocess_vegetation.py
   (It will now use combined_disturbance_year.tif if available)

Or manually calculate forest age:
   forest_age = 2024 - disturbance_year
   (Use 100 years for pixels with no recorded disturbance)
""")


if __name__ == "__main__":
    main()
