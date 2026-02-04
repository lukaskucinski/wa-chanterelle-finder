"""
Script 03: Preprocess vegetation data.

This script processes:
1. LANDFIRE Existing Vegetation Type (EVT) - forest species
2. NLCD Tree Canopy Cover - canopy density
3. Forest age (if available from LANDFIRE VDIST)

Output:
- Forest type classification raster
- Canopy cover percentage raster
- Forest age raster (optional)
"""

import os
import sys
from pathlib import Path

import numpy as np
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
    reproject_raster,
    clip_raster_to_bounds,
    align_raster_to_template,
    get_raster_stats,
)

# Directories
RAW_VEG_DIR = PROJECT_ROOT / "data" / "raw" / "vegetation"
RAW_CANOPY_DIR = PROJECT_ROOT / "data" / "raw" / "canopy"
RAW_HDIST_DIR = PROJECT_ROOT / "data" / "raw" / "vegetation" / "landfire_hdist"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Current year for calculating time since disturbance
CURRENT_YEAR = 2024

# LANDFIRE EVT codes for Pacific Northwest forests
# Reference: https://landfire.gov/evt.php
# These codes identify forest types with chanterelle mycorrhizal associations

# Pacific Northwest Conifer Forest Types (approximate codes)
# Note: Actual codes should be verified from LANDFIRE documentation
EVT_DOUGLAS_FIR = {
    7051, 7052, 7053, 7054, 7055,  # Douglas-fir variants
    7112, 7113, 7114,  # Douglas-fir / Western Hemlock mix
    7125, 7126,  # Douglas-fir / Grand Fir
}

EVT_WESTERN_HEMLOCK = {
    7061, 7062, 7063, 7064,  # Western Hemlock variants
    7071, 7072, 7073,  # Mountain Hemlock variants
}

EVT_SPRUCE = {
    7081, 7082, 7083,  # Sitka Spruce variants
    7091, 7092, 7093,  # Engelmann Spruce variants
}

EVT_OTHER_CONIFER = {
    7021, 7022, 7023,  # Western Red Cedar
    7031, 7032, 7033,  # Pacific Silver Fir
    7041, 7042, 7043,  # Grand Fir
    7101, 7102, 7103,  # Noble Fir
    7141, 7142, 7143,  # Western White Pine
    7151, 7152, 7153,  # Lodgepole Pine
    7161, 7162, 7163,  # Ponderosa Pine (drier, less suitable)
}

# Optimal chanterelle forest types (primary mycorrhizal hosts)
EVT_OPTIMAL_CHANTERELLE = EVT_DOUGLAS_FIR | EVT_WESTERN_HEMLOCK | EVT_SPRUCE

# Marginal forest types (some chanterelle potential)
EVT_MARGINAL_CHANTERELLE = EVT_OTHER_CONIFER


def classify_forest_type(evt_code: np.ndarray) -> np.ndarray:
    """
    Classify forest type for chanterelle suitability.

    Returns:
        Classified array:
        - 3: Optimal (Douglas-fir, Western Hemlock, Spruce)
        - 2: Marginal (Other conifers)
        - 1: Poor (Mixed/hardwood)
        - 0: Non-forest
    """
    classified = np.zeros_like(evt_code, dtype=np.uint8)

    # Optimal chanterelle habitat
    for code in EVT_OPTIMAL_CHANTERELLE:
        classified[evt_code == code] = 3

    # Marginal habitat
    for code in EVT_MARGINAL_CHANTERELLE:
        # Only set if not already optimal
        mask = (evt_code == code) & (classified == 0)
        classified[mask] = 2

    # Any other forest (codes typically 7000-7999 for forest)
    forest_mask = (evt_code >= 7000) & (evt_code < 8000) & (classified == 0)
    classified[forest_mask] = 1

    return classified


def process_vegetation():
    """Main vegetation processing workflow."""
    print("=" * 60)
    print("VEGETATION PREPROCESSING")
    print("=" * 60)

    # Check for template (DEM must be processed first)
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    if not template_path.exists():
        print("ERROR: Template raster not found. Run 02_preprocess_dem.py first.")
        return

    # Output paths
    evt_clipped_path = PROCESSED_DIR / "landfire_evt_clipped.tif"
    evt_aligned_path = PROCESSED_DIR / "landfire_evt_aligned.tif"
    forest_type_path = PROCESSED_DIR / "forest_type.tif"
    canopy_aligned_path = PROCESSED_DIR / "canopy_cover.tif"

    # Find input files
    evt_files = list(RAW_VEG_DIR.glob("*EVT*.tif")) + list(RAW_VEG_DIR.glob("*evt*.tif"))
    canopy_files = list(RAW_CANOPY_DIR.glob("*tcc*.tif")) + list(RAW_CANOPY_DIR.glob("*canopy*.tif"))

    # Process LANDFIRE EVT
    print("\n[1/3] Processing LANDFIRE EVT (Existing Vegetation Type)...")

    if evt_files:
        evt_input = evt_files[0]
        print(f"Input: {evt_input}")

        # Clip to study area
        if not evt_clipped_path.exists():
            print("Clipping to study area...")
            clip_raster_to_bounds(evt_input, evt_clipped_path)

        # Align to template
        if not evt_aligned_path.exists():
            print("Aligning to template grid...")
            align_raster_to_template(
                evt_clipped_path,
                template_path,
                evt_aligned_path,
                resampling=Resampling.nearest  # Categorical data
            )

        # Classify forest types
        if not forest_type_path.exists():
            print("Classifying forest types...")
            evt_data, meta = read_raster(evt_aligned_path)

            # Handle nodata
            nodata = meta['nodata']
            if nodata is not None:
                evt_data = np.where(evt_data == nodata, 0, evt_data)

            forest_class = classify_forest_type(evt_data.astype(np.int32))
            write_raster(
                forest_type_path,
                forest_class,
                meta['transform'],
                TARGET_CRS,
                nodata=255,
                dtype='uint8'
            )
            print(f"Forest type saved to: {forest_type_path}")

            # Print statistics
            unique, counts = np.unique(forest_class, return_counts=True)
            print("\nForest Type Distribution:")
            labels = {0: "Non-forest", 1: "Other forest", 2: "Marginal conifer", 3: "Optimal conifer"}
            for val, count in zip(unique, counts):
                pct = count / forest_class.size * 100
                print(f"  {labels.get(val, val)}: {count:,} cells ({pct:.1f}%)")
    else:
        print(f"WARNING: No LANDFIRE EVT files found in {RAW_VEG_DIR}")
        print("Creating placeholder forest type raster...")

        # Create placeholder with all optimal habitat (for testing)
        template_data, meta = read_raster(template_path)
        placeholder = np.ones_like(template_data, dtype=np.uint8) * 3
        placeholder[np.isnan(template_data) | (template_data == NODATA)] = 0
        write_raster(forest_type_path, placeholder, meta['transform'], TARGET_CRS, nodata=255, dtype='uint8')
        print(f"Placeholder forest type saved to: {forest_type_path}")

    # Process Canopy Cover
    print("\n[2/3] Processing NLCD Tree Canopy Cover...")

    if canopy_files:
        canopy_input = canopy_files[0]
        print(f"Input: {canopy_input}")

        if not canopy_aligned_path.exists():
            print("Aligning canopy cover to template grid...")

            # First clip
            canopy_clipped = PROCESSED_DIR / "canopy_clipped.tif"
            if not canopy_clipped.exists():
                clip_raster_to_bounds(canopy_input, canopy_clipped)

            # Then align
            align_raster_to_template(
                canopy_clipped,
                template_path,
                canopy_aligned_path,
                resampling=Resampling.bilinear
            )

            # Mask invalid NLCD values (254, 255 are fill/nodata)
            canopy_data, canopy_meta = read_raster(canopy_aligned_path)
            canopy_data = np.where(canopy_data > 100, NODATA, canopy_data)
            write_raster(canopy_aligned_path, canopy_data, canopy_meta['transform'], TARGET_CRS)
            print(f"Canopy cover saved to: {canopy_aligned_path}")

            # Print statistics
            stats = get_raster_stats(canopy_aligned_path)
            print(f"\nCanopy Cover Statistics:")
            print(f"  Min: {stats['min']:.1f}%")
            print(f"  Max: {stats['max']:.1f}%")
            print(f"  Mean: {stats['mean']:.1f}%")
    else:
        print(f"WARNING: No canopy cover files found in {RAW_CANOPY_DIR}")
        print("Creating placeholder canopy cover raster...")

        # Create placeholder with moderate canopy (for testing)
        template_data, meta = read_raster(template_path)
        placeholder = np.ones_like(template_data, dtype=np.float32) * 60  # 60% canopy
        placeholder[np.isnan(template_data) | (template_data == NODATA)] = NODATA
        write_raster(canopy_aligned_path, placeholder, meta['transform'], TARGET_CRS)
        print(f"Placeholder canopy cover saved to: {canopy_aligned_path}")

    # Forest Age (if available)
    print("\n[3/3] Processing Forest Age (optional)...")
    forest_age_path = PROCESSED_DIR / "forest_age.tif"

    # Look for disturbance data in order of preference:
    # 1. Combined Hansen + LANDFIRE (most complete)
    # 2. LANDFIRE HDist (Historical Disturbance Year)
    # 3. Our processed "last disturbance year" from annual data
    # 4. VDIST files

    # Check for combined disturbance data first (Hansen + LANDFIRE)
    combined_dist = PROCESSED_DIR / "combined_disturbance_year.tif"
    hansen_dist = PROCESSED_DIR / "hansen_loss_year.tif"

    hdist_files = []
    if combined_dist.exists():
        hdist_files = [combined_dist]
        print(f"  Using combined Hansen + LANDFIRE disturbance data")
    elif hansen_dist.exists():
        hdist_files = [hansen_dist]
        print(f"  Using Hansen forest loss data")
    else:
        # Fall back to LANDFIRE data
        hdist_files = list(RAW_HDIST_DIR.glob("*HDist*.tif")) + list(RAW_HDIST_DIR.glob("*hdist*.tif"))
        hdist_files += list(RAW_HDIST_DIR.glob("*HDIST*.tif"))
        # Also check for our processed "last disturbance year" file
        hdist_files += list(RAW_HDIST_DIR.glob("*last_disturbance*.tif"))
        hdist_files += list(RAW_HDIST_DIR.glob("*disturbance_year*.tif"))

    vdist_files = list(RAW_VEG_DIR.glob("*VDIST*.tif")) + list(RAW_VEG_DIR.glob("*vdist*.tif"))

    if hdist_files and not forest_age_path.exists():
        # HDist contains year of last disturbance - calculate time since
        hdist_input = hdist_files[0]
        print(f"Input (HDist): {hdist_input}")

        print("Processing historical disturbance data...")

        # First align to template
        hdist_aligned = PROCESSED_DIR / "hdist_aligned.tif"
        if not hdist_aligned.exists():
            align_raster_to_template(
                hdist_input,
                template_path,
                hdist_aligned,
                resampling=Resampling.nearest
            )

        # Read and calculate time since disturbance
        hdist_data, meta = read_raster(hdist_aligned)

        # HDist values are years (e.g., 2015, 2020) or 0/NoData for no disturbance
        # Calculate age = Current Year - Disturbance Year
        # For areas with no recorded disturbance, assume old-growth (100+ years)
        valid_years = (hdist_data >= 1900) & (hdist_data <= CURRENT_YEAR)

        forest_age = np.full_like(hdist_data, NODATA, dtype=np.float32)
        forest_age[valid_years] = CURRENT_YEAR - hdist_data[valid_years]

        # No disturbance recorded = assume old-growth (use 100 years)
        no_disturbance = (hdist_data == 0) | (hdist_data == -9999) | np.isnan(hdist_data)
        template_data, _ = read_raster(template_path)
        valid_mask = ~np.isnan(template_data) & (template_data != NODATA)
        forest_age[no_disturbance & valid_mask] = 100

        write_raster(forest_age_path, forest_age, meta['transform'], TARGET_CRS)
        print(f"Forest age (from HDist) saved to: {forest_age_path}")

        # Statistics
        valid_age = forest_age[(forest_age != NODATA) & (forest_age >= 0)]
        if len(valid_age) > 0:
            print(f"\nForest Age Statistics:")
            print(f"  Min: {valid_age.min():.0f} years")
            print(f"  Max: {valid_age.max():.0f} years")
            print(f"  Mean: {valid_age.mean():.0f} years")

    elif vdist_files and not forest_age_path.exists():
        vdist_input = vdist_files[0]
        print(f"Input (VDist): {vdist_input}")

        print("Processing vegetation disturbance data...")
        align_raster_to_template(
            vdist_input,
            template_path,
            forest_age_path,
            resampling=Resampling.nearest
        )
        print(f"Forest age saved to: {forest_age_path}")

    elif not forest_age_path.exists():
        print("No VDIST/HDist data found. Creating estimated forest age layer...")

        # Create placeholder based on forest type
        # Assume optimal conifer = second-growth (60 years), other = mixed ages
        if forest_type_path.exists():
            forest_type, meta = read_raster(forest_type_path)

            age = np.zeros_like(forest_type, dtype=np.float32)
            age[forest_type == 3] = 60  # Optimal conifer = second-growth
            age[forest_type == 2] = 80  # Marginal = mature
            age[forest_type == 1] = 40  # Other forest = younger
            age[forest_type == 0] = NODATA

            write_raster(forest_age_path, age, meta['transform'], TARGET_CRS)
            print(f"Estimated forest age saved to: {forest_age_path}")
        else:
            print("Cannot create forest age without forest type data")

    print("\n" + "=" * 60)
    print("Vegetation preprocessing complete!")
    print("=" * 60)


def main():
    """Entry point."""
    process_vegetation()


if __name__ == "__main__":
    main()
