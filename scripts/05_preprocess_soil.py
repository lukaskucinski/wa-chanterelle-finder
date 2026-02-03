"""
Script 05: Preprocess soil data from SSURGO/gSSURGO.

This script processes:
1. Soil pH (1:1 water method)
2. Soil drainage class

SSURGO data is complex - it comes as linked database tables.
This script handles both:
- Rasterized gSSURGO GeoTIFF
- Vector shapefile with attributes

Output:
- Soil pH raster
- Soil drainage class raster
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
import rasterio
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
    rasterize_vector,
    align_raster_to_template,
    get_raster_stats,
)

# Directories
RAW_SOIL_DIR = PROJECT_ROOT / "data" / "raw" / "soil"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Soil drainage class codes (SSURGO drainagecl field)
DRAINAGE_CODES = {
    'Excessively drained': 1,
    'Somewhat excessively drained': 2,
    'Well drained': 3,
    'Moderately well drained': 4,
    'Somewhat poorly drained': 5,
    'Poorly drained': 6,
    'Very poorly drained': 7,
}

# Chanterelle-suitable drainage (well-drained to moderately well-drained)
OPTIMAL_DRAINAGE = {1, 2, 3, 4}


def find_soil_files(soil_dir: Path) -> dict:
    """
    Find soil data files in the directory.

    Returns dict with paths to various soil data formats.
    """
    files = {}

    # Check for gSSURGO raster
    for pattern in ["*gSSURGO*.tif", "*gssurgo*.tif", "*MapunitRaster*.tif"]:
        matches = list(soil_dir.glob(pattern))
        if matches:
            files['gssurgo_raster'] = matches[0]
            break

    # Check for gSSURGO geodatabase
    for pattern in ["*.gdb", "gSSURGO*.gdb"]:
        for gdb in soil_dir.glob(pattern):
            if gdb.is_dir():
                files['gssurgo_gdb'] = gdb
                break

    # Check for SSURGO shapefiles
    for pattern in ["*soilmu*.shp", "*MUPOLYGON*.shp", "*mupolygon*.shp"]:
        matches = list(soil_dir.glob(pattern))
        if matches:
            files['mupolygon'] = matches[0]
            break

    # Check for tabular data
    for pattern in ["*chorizon*.csv", "*chorizon*.txt", "chorizon.csv"]:
        matches = list(soil_dir.rglob(pattern))
        if matches:
            files['chorizon'] = matches[0]
            break

    for pattern in ["*component*.csv", "*component*.txt", "component.csv"]:
        matches = list(soil_dir.rglob(pattern))
        if matches:
            files['component'] = matches[0]
            break

    return files


def process_ssurgo_vector(
    mupolygon_path: Path,
    template_path: Path,
    chorizon_path: Optional[Path] = None,
    component_path: Optional[Path] = None
) -> tuple:
    """
    Process SSURGO vector data to extract pH and drainage.

    This is a simplified approach - full SSURGO processing
    requires joining multiple tables.
    """
    print(f"Reading soil polygons: {mupolygon_path}")
    gdf = gpd.read_file(mupolygon_path)

    # Reproject if needed
    if gdf.crs != TARGET_CRS:
        print("Reprojecting to UTM...")
        gdf = gdf.to_crs(TARGET_CRS)

    # Check for pH and drainage attributes
    # SSURGO field names vary - check common ones
    ph_fields = ['ph1to1h2o_r', 'ph1to1h2o', 'ph_r', 'pH', 'ph']
    drainage_fields = ['drainagecl', 'drainage', 'Drainage']

    ph_field = None
    drainage_field = None

    for field in ph_fields:
        if field in gdf.columns:
            ph_field = field
            break

    for field in drainage_fields:
        if field in gdf.columns:
            drainage_field = field
            break

    print(f"Found fields - pH: {ph_field}, Drainage: {drainage_field}")

    # Rasterize pH
    ph_raster = None
    if ph_field:
        print("Rasterizing soil pH...")
        gdf['ph_numeric'] = pd.to_numeric(gdf[ph_field], errors='coerce')
        ph_raster = rasterize_vector(gdf, template_path, attribute='ph_numeric', fill_value=NODATA)

    # Rasterize drainage
    drainage_raster = None
    if drainage_field:
        print("Rasterizing drainage class...")
        # Convert text drainage to numeric code
        gdf['drainage_code'] = gdf[drainage_field].map(DRAINAGE_CODES).fillna(0).astype(int)
        drainage_raster = rasterize_vector(gdf, template_path, attribute='drainage_code', fill_value=0)

    return ph_raster, drainage_raster


def create_default_soil_layers(template_path: Path) -> tuple:
    """
    Create default soil layers when no SSURGO data is available.

    Uses typical values for Cascades volcanic soils:
    - pH: ~5.0-5.5 (acidic, typical for conifer forests)
    - Drainage: Well-drained (volcanic soils)
    """
    print("Creating default soil layers based on typical Cascades conditions...")

    template_data, meta = read_raster(template_path)
    valid_mask = ~np.isnan(template_data) & (template_data != NODATA)

    # Default pH: slightly acidic, with variation by elevation
    # Lower elevations tend to have slightly lower pH
    elevation = template_data.copy()
    elevation[~valid_mask] = 0

    # pH range 4.5-5.5, higher elevation = slightly higher pH
    ph_base = 4.8
    ph_variation = (elevation - elevation[valid_mask].min()) / (elevation[valid_mask].max() - elevation[valid_mask].min() + 1)
    ph_raster = np.where(valid_mask, ph_base + ph_variation * 0.7, NODATA)

    # Default drainage: well-drained (code 3) for most areas
    # Volcanic soils in Cascades are typically well-drained
    drainage_raster = np.where(valid_mask, 3, 0).astype(np.float32)

    return ph_raster.astype(np.float32), drainage_raster


def process_soil():
    """Main soil processing workflow."""
    print("=" * 60)
    print("SOIL PREPROCESSING")
    print("=" * 60)

    # Check for template
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    if not template_path.exists():
        print("ERROR: Template raster not found. Run 02_preprocess_dem.py first.")
        return

    # Output paths
    ph_path = PROCESSED_DIR / "soil_ph.tif"
    drainage_path = PROCESSED_DIR / "soil_drainage.tif"

    # Find soil data
    soil_files = find_soil_files(RAW_SOIL_DIR)
    print(f"\nFound soil data: {list(soil_files.keys())}")

    ph_raster = None
    drainage_raster = None

    # Try different data sources
    if 'gssurgo_raster' in soil_files:
        print("\n[1/2] Processing gSSURGO raster...")
        # gSSURGO raster contains MUKEY values - need to join with tables
        # For simplicity, we'll use default values unless full processing is implemented
        print("gSSURGO raster processing requires table joins (not implemented)")
        print("Using default soil values...")

    if 'mupolygon' in soil_files and (ph_raster is None or drainage_raster is None):
        print("\n[1/2] Processing SSURGO vector polygons...")
        try:
            import pandas as pd
            ph_raster, drainage_raster = process_ssurgo_vector(
                soil_files['mupolygon'],
                template_path,
                soil_files.get('chorizon'),
                soil_files.get('component')
            )
        except Exception as e:
            print(f"Error processing SSURGO: {e}")
            print("Using default soil values...")

    # Fall back to defaults if needed
    if ph_raster is None or drainage_raster is None:
        print("\nNo complete soil data found. Using default values.")
        ph_default, drainage_default = create_default_soil_layers(template_path)

        if ph_raster is None:
            ph_raster = ph_default
        if drainage_raster is None:
            drainage_raster = drainage_default

    # Get metadata from template
    _, meta = read_raster(template_path)

    # Save pH raster
    print("\n[2/2] Saving soil layers...")

    if not ph_path.exists():
        write_raster(ph_path, ph_raster, meta['transform'], TARGET_CRS)
        print(f"Soil pH saved to: {ph_path}")

        stats = get_raster_stats(ph_path)
        print(f"\nSoil pH Statistics:")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Mean: {stats['mean']:.2f}")

    if not drainage_path.exists():
        write_raster(drainage_path, drainage_raster, meta['transform'], TARGET_CRS)
        print(f"Soil drainage saved to: {drainage_path}")

        if drainage_raster is not None:
            unique, counts = np.unique(drainage_raster[drainage_raster > 0], return_counts=True)
            print(f"\nDrainage Class Distribution:")
            drainage_names = {v: k for k, v in DRAINAGE_CODES.items()}
            for val, count in zip(unique, counts):
                name = drainage_names.get(int(val), f"Code {int(val)}")
                pct = count / np.sum(counts) * 100
                print(f"  {name}: {count:,} cells ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("Soil preprocessing complete!")
    print("=" * 60)


def main():
    """Entry point."""
    process_soil()


if __name__ == "__main__":
    main()
