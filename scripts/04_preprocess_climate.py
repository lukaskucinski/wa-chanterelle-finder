"""
Script 04: Preprocess climate data.

This script processes PRISM precipitation data:
1. Annual precipitation normals
2. Fall precipitation (Sep-Dec) for chanterelle season

Output:
- Annual precipitation (inches)
- Fall precipitation (inches)
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
    clip_raster_to_bounds,
    align_raster_to_template,
    get_raster_stats,
)

# Directories
RAW_CLIMATE_DIR = PROJECT_ROOT / "data" / "raw" / "climate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def find_prism_files(climate_dir: Path) -> dict:
    """
    Find PRISM precipitation files in the climate directory.

    Returns dict with keys: 'annual', 'sep', 'oct', 'nov', 'dec'
    """
    files = {}

    # Search patterns for PRISM files (recursive to handle subdirectory structure)
    # Files may be in subdirectories like: prism_ppt_us_30s_2020_avg_30y/prism_ppt_us_30s_2020_avg_30y.tif

    for f in climate_dir.glob("**/*ppt*.tif"):
        name = f.name.lower()

        if '2020_avg' in name or 'annual' in name:
            files['annual'] = f
        elif '202009' in name or '_09_' in name:
            files['sep'] = f
        elif '202010' in name or '_10_' in name:
            files['oct'] = f
        elif '202011' in name or '_11_' in name:
            files['nov'] = f
        elif '202012' in name or '_12_' in name:
            files['dec'] = f

    return files


def mm_to_inches(mm: np.ndarray) -> np.ndarray:
    """Convert millimeters to inches."""
    return mm / 25.4


def process_climate():
    """Main climate processing workflow."""
    print("=" * 60)
    print("CLIMATE PREPROCESSING")
    print("=" * 60)

    # Check for template
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    if not template_path.exists():
        print("ERROR: Template raster not found. Run 02_preprocess_dem.py first.")
        return

    # Output paths
    annual_precip_path = PROCESSED_DIR / "annual_precip_inches.tif"
    fall_precip_path = PROCESSED_DIR / "fall_precip_inches.tif"

    # Find PRISM files
    prism_files = find_prism_files(RAW_CLIMATE_DIR)
    print(f"\nFound PRISM files: {list(prism_files.keys())}")

    # Process Annual Precipitation
    print("\n[1/2] Processing Annual Precipitation...")

    if 'annual' in prism_files:
        annual_input = prism_files['annual']
        print(f"Input: {annual_input}")

        if not annual_precip_path.exists():
            # Clip and align
            annual_clipped = PROCESSED_DIR / "annual_precip_clipped.tif"
            if not annual_clipped.exists():
                print("Clipping to study area...")
                clip_raster_to_bounds(annual_input, annual_clipped)

            print("Aligning to template grid...")
            annual_aligned = PROCESSED_DIR / "annual_precip_aligned.tif"
            align_raster_to_template(
                annual_clipped,
                template_path,
                annual_aligned,
                resampling=Resampling.bilinear
            )

            # Convert mm to inches (PRISM data is in mm)
            print("Converting mm to inches...")
            precip_data, meta = read_raster(annual_aligned)

            # Handle nodata
            nodata = meta['nodata']
            valid_mask = ~np.isnan(precip_data)
            if nodata is not None:
                valid_mask &= (precip_data != nodata)

            precip_inches = np.where(valid_mask, mm_to_inches(precip_data), NODATA)

            write_raster(annual_precip_path, precip_inches, meta['transform'], TARGET_CRS)
            print(f"Annual precipitation saved to: {annual_precip_path}")

            # Statistics
            stats = get_raster_stats(annual_precip_path)
            print(f"\nAnnual Precipitation Statistics:")
            print(f"  Min: {stats['min']:.1f} inches")
            print(f"  Max: {stats['max']:.1f} inches")
            print(f"  Mean: {stats['mean']:.1f} inches")
    else:
        print(f"WARNING: No annual precipitation file found in {RAW_CLIMATE_DIR}")
        print("Creating placeholder based on typical Cascades precipitation...")

        # Create placeholder with gradient (wetter on west side)
        template_data, meta = read_raster(template_path)
        valid_mask = ~np.isnan(template_data) & (template_data != NODATA)

        # Simple east-west gradient (60-120 inches)
        height, width = template_data.shape
        gradient = np.linspace(100, 50, width)  # West to east
        gradient = np.tile(gradient, (height, 1))

        # Add some elevation influence (higher = more precip up to a point)
        elev_factor = np.clip(template_data / 5000, 0, 1) * 20  # Up to 20" extra

        placeholder = np.where(valid_mask, gradient + elev_factor, NODATA)
        write_raster(annual_precip_path, placeholder.astype(np.float32), meta['transform'], TARGET_CRS)
        print(f"Placeholder annual precipitation saved to: {annual_precip_path}")

    # Process Fall Precipitation (Sep-Dec)
    print("\n[2/2] Processing Fall Precipitation (Chanterelle Season)...")

    # Require Sep-Nov, optionally include Dec if available
    if all(m in prism_files for m in ['sep', 'oct', 'nov']):
        fall_months = ['sep', 'oct', 'nov']
        if 'dec' in prism_files:
            fall_months.append('dec')
        print(f"Processing monthly files ({', '.join(fall_months)})...")

        fall_total = None

        for month in fall_months:
            month_input = prism_files[month]
            print(f"  Processing {month}: {month_input.name}")

            # Clip and align
            month_clipped = PROCESSED_DIR / f"{month}_precip_clipped.tif"
            month_aligned = PROCESSED_DIR / f"{month}_precip_aligned.tif"

            if not month_clipped.exists():
                clip_raster_to_bounds(month_input, month_clipped)

            if not month_aligned.exists():
                align_raster_to_template(
                    month_clipped,
                    template_path,
                    month_aligned,
                    resampling=Resampling.bilinear
                )

            # Read and accumulate
            month_data, meta = read_raster(month_aligned)

            if fall_total is None:
                fall_total = np.zeros_like(month_data, dtype=np.float64)
                valid_mask = ~np.isnan(month_data)
                if meta['nodata'] is not None:
                    valid_mask &= (month_data != meta['nodata'])

            # Add to total (only where valid)
            month_valid = ~np.isnan(month_data)
            if meta['nodata'] is not None:
                month_valid &= (month_data != meta['nodata'])

            fall_total = np.where(month_valid, fall_total + month_data, fall_total)

        # Convert to inches and save
        fall_inches = np.where(valid_mask, mm_to_inches(fall_total), NODATA)
        write_raster(fall_precip_path, fall_inches.astype(np.float32), meta['transform'], TARGET_CRS)
        print(f"Fall precipitation saved to: {fall_precip_path}")

        stats = get_raster_stats(fall_precip_path)
        month_range = "Sep-Dec" if 'dec' in fall_months else "Sep-Nov"
        print(f"\nFall Precipitation ({month_range}) Statistics:")
        print(f"  Min: {stats['min']:.1f} inches")
        print(f"  Max: {stats['max']:.1f} inches")
        print(f"  Mean: {stats['mean']:.1f} inches")

    else:
        print("Monthly files not found. Estimating fall precipitation from annual...")

        # Estimate fall as ~40% of annual (typical for PNW)
        if annual_precip_path.exists():
            annual_data, meta = read_raster(annual_precip_path)
            valid_mask = ~np.isnan(annual_data) & (annual_data != NODATA)

            fall_precip = np.where(valid_mask, annual_data * 0.4, NODATA)
            write_raster(fall_precip_path, fall_precip.astype(np.float32), meta['transform'], TARGET_CRS)
            print(f"Estimated fall precipitation saved to: {fall_precip_path}")

            stats = get_raster_stats(fall_precip_path)
            print(f"\nFall Precipitation (estimated) Statistics:")
            print(f"  Min: {stats['min']:.1f} inches")
            print(f"  Max: {stats['max']:.1f} inches")
            print(f"  Mean: {stats['mean']:.1f} inches")
        else:
            print("Cannot create fall precipitation without annual data")

    print("\n" + "=" * 60)
    print("Climate preprocessing complete!")
    print("=" * 60)


def main():
    """Entry point."""
    process_climate()


if __name__ == "__main__":
    main()
