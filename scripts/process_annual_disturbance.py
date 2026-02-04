"""
Process LANDFIRE Annual Disturbance data to create a "Year of Last Disturbance" layer.

Since HDist (Historical Disturbance) requires a helpdesk request, we can create
an equivalent layer by processing the annual disturbance files (1999-present).

For each pixel, we find the most recent year it was disturbed.
Forest Age = Current Year - Last Disturbance Year

Input: All Years Annual Disturbance zip file containing nested zips for each year
Output: Single raster with year of last disturbance per pixel
"""

import os
import sys
import zipfile
from pathlib import Path
from glob import glob

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.raster_utils import align_raster_to_template

# Directories
RAW_HDIST_DIR = PROJECT_ROOT / "data" / "raw" / "vegetation" / "landfire_hdist"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEMP_DIR = RAW_HDIST_DIR / "extracted"

# Study area bounds (approximate WA Cascades in WGS84)
# Will be used to clip CONUS data to reduce memory usage
WA_BOUNDS = {
    'west': -125.0,
    'east': -117.0,
    'south': 45.5,
    'north': 49.5
}

CURRENT_YEAR = 2024


def extract_nested_zips(input_dir: Path, output_dir: Path):
    """Extract all nested zip files from the LANDFIRE download."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all zip files
    zip_files = list(input_dir.glob("*.zip"))

    # Also check for the main downloaded zip
    main_zips = list(input_dir.glob("*Disturbance*.zip")) + list(input_dir.glob("*Annual*.zip"))

    print(f"Found {len(zip_files)} zip files in {input_dir}")

    extracted_tifs = []

    for zip_path in sorted(zip_files):
        print(f"  Extracting: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # List contents
                names = zf.namelist()
                tif_files = [n for n in names if n.endswith('.tif')]

                if tif_files:
                    # Extract TIF files
                    for tif in tif_files:
                        zf.extract(tif, output_dir)
                        extracted_tifs.append(output_dir / tif)
                        print(f"    -> {tif}")
                else:
                    # May contain more zips - extract all
                    zf.extractall(output_dir)

        except zipfile.BadZipFile:
            print(f"    WARNING: Bad zip file, skipping")
            continue

    # Check for any nested zips that were extracted
    nested_zips = list(output_dir.glob("**/*.zip"))
    for nested_zip in nested_zips:
        print(f"  Extracting nested: {nested_zip.name}")
        try:
            with zipfile.ZipFile(nested_zip, 'r') as zf:
                zf.extractall(nested_zip.parent)
        except:
            pass

    return extracted_tifs


def find_disturbance_tifs(search_dir: Path) -> dict:
    """Find all annual disturbance TIF files and map to years."""
    year_files = {}

    # Search patterns for different naming conventions
    for tif_path in search_dir.glob("**/*.tif"):
        name = tif_path.name.upper()

        # Try to extract year from filename
        year = None

        # Pattern 1: us_dist1999.tif, us_dist2014.tif
        if "DIST19" in name or "DIST20" in name:
            try:
                idx = name.find("DIST")
                year = int(name[idx+4:idx+8])
            except:
                pass

        # Pattern 2: LC15_Dist_200.tif (year = 2015), LC24_Dist_250.tif (year = 2024)
        elif name.startswith("LC") and "_DIST" in name:
            try:
                # LC15 -> 2015, LC24 -> 2024
                yy = int(name[2:4])
                year = 2000 + yy if yy < 50 else 1900 + yy
            except:
                pass

        # Pattern 3: LF2024_Dist..., LF2023_Dist...
        elif "LF20" in name:
            try:
                idx = name.find("LF20")
                year = int(name[idx+2:idx+6])
            except:
                pass

        # Pattern 4: Generic year search
        if year is None:
            for y in range(1999, 2026):
                if str(y) in name:
                    year = y
                    break

        if year and 1999 <= year <= 2025:
            if year not in year_files:
                year_files[year] = tif_path
                print(f"  Found year {year}: {tif_path.name}")

    return year_files


def read_clipped_raster(tif_path: Path, bounds: dict) -> tuple:
    """Read a raster clipped to bounds to save memory."""
    from rasterio.crs import CRS
    from pyproj import Transformer

    with rasterio.open(tif_path) as src:
        try:
            # Transform WGS84 bounds to raster CRS
            src_crs = src.crs
            wgs84 = CRS.from_epsg(4326)

            if src_crs != wgs84:
                # Transform bounds from WGS84 to raster CRS
                transformer = Transformer.from_crs(wgs84, src_crs, always_xy=True)
                west, south = transformer.transform(bounds['west'], bounds['south'])
                east, north = transformer.transform(bounds['east'], bounds['north'])
            else:
                west, south = bounds['west'], bounds['south']
                east, north = bounds['east'], bounds['north']

            # Calculate window from transformed bounds
            window = from_bounds(west, south, east, north, src.transform)

            # Ensure window is valid
            window = window.round_offsets().round_lengths()

            if window.width <= 0 or window.height <= 0:
                print(f"    WARNING: Invalid window for {tif_path.name}")
                return None, None, None, None

            # Read windowed data
            data = src.read(1, window=window)

            # Calculate new transform for the window
            transform = src.window_transform(window)

            return data, transform, src.crs, src.nodata

        except Exception as e:
            print(f"    Error reading {tif_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None


def create_last_disturbance_year(year_files: dict, output_path: Path):
    """
    Create a composite raster showing the most recent disturbance year per pixel.
    """
    print(f"\nCreating last disturbance year composite from {len(year_files)} years...")

    # Sort years
    years = sorted(year_files.keys())
    print(f"  Years: {years[0]} to {years[-1]}")

    # Read first file to get dimensions and metadata
    first_year = years[0]
    first_data, transform, crs, nodata = read_clipped_raster(
        year_files[first_year], WA_BOUNDS
    )

    if first_data is None:
        print("ERROR: Could not read first raster")
        return None

    print(f"  Clipped raster shape: {first_data.shape}")

    # Initialize output array (year of last disturbance)
    # Start with 0 (no disturbance recorded)
    last_dist_year = np.zeros(first_data.shape, dtype=np.int16)

    # Process each year
    for year in years:
        print(f"  Processing {year}...", end=" ", flush=True)

        data, _, _, nd = read_clipped_raster(year_files[year], WA_BOUNDS)

        if data is None:
            print("SKIP")
            continue

        # Disturbance is typically indicated by non-zero values
        # Update last_dist_year where this year had disturbance
        disturbed = (data > 0)
        if nd is not None:
            disturbed &= (data != nd)

        count = np.sum(disturbed)
        last_dist_year[disturbed] = year

        print(f"({count:,} pixels disturbed)")

    # Save output
    print(f"\nSaving to: {output_path}")

    profile = {
        'driver': 'GTiff',
        'dtype': 'int16',
        'width': last_dist_year.shape[1],
        'height': last_dist_year.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'nodata': 0,
        'compress': 'lzw'
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(last_dist_year, 1)

    # Statistics
    valid = last_dist_year > 0
    if np.any(valid):
        print(f"\nLast Disturbance Year Statistics:")
        print(f"  Pixels with disturbance: {np.sum(valid):,}")
        print(f"  Min year: {last_dist_year[valid].min()}")
        print(f"  Max year: {last_dist_year[valid].max()}")
        print(f"  Pixels with no recorded disturbance: {np.sum(~valid):,}")

    return output_path


def main():
    """Main processing workflow."""
    print("=" * 60)
    print("LANDFIRE Annual Disturbance Processing")
    print("=" * 60)

    # Check for input data
    if not RAW_HDIST_DIR.exists():
        print(f"ERROR: Input directory not found: {RAW_HDIST_DIR}")
        print("Please extract the LANDFIRE disturbance data first.")
        return

    # Output path
    output_path = RAW_HDIST_DIR / "last_disturbance_year_WA.tif"

    if output_path.exists():
        print(f"Output already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return

    # Step 1: Extract nested zips if needed
    print("\n[1/3] Checking for nested zip files...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we have direct TIFs or need to extract
    existing_tifs = list(RAW_HDIST_DIR.glob("**/*.tif"))
    if len(existing_tifs) < 10:  # Expect ~25 years of data
        print("Extracting zip files...")
        extract_nested_zips(RAW_HDIST_DIR, TEMP_DIR)
    else:
        print(f"Found {len(existing_tifs)} existing TIF files")

    # Step 2: Find all annual TIFs
    print("\n[2/3] Finding annual disturbance files...")
    year_files = find_disturbance_tifs(RAW_HDIST_DIR)

    if not year_files:
        year_files = find_disturbance_tifs(TEMP_DIR)

    if len(year_files) < 5:
        print(f"ERROR: Only found {len(year_files)} year files. Expected 20+")
        print("Please ensure the LANDFIRE data is properly extracted.")
        return

    print(f"\nFound {len(year_files)} years of disturbance data")

    # Step 3: Create composite
    print("\n[3/4] Creating last disturbance year composite...")
    create_last_disturbance_year(year_files, output_path)

    # Step 4: Align to template grid
    print("\n[4/4] Aligning to project template grid...")
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    aligned_path = PROCESSED_DIR / "landfire_disturbance_year.tif"

    if template_path.exists():
        print(f"  Template: {template_path}")
        print(f"  Output: {aligned_path}")

        align_raster_to_template(
            output_path,
            template_path,
            aligned_path,
            resampling=Resampling.nearest  # Categorical data (years)
        )

        # Print aligned statistics
        with rasterio.open(aligned_path) as src:
            data = src.read(1)
            valid = data > 0
            if np.any(valid):
                print(f"\n  Aligned Disturbance Statistics:")
                print(f"    Shape: {data.shape}")
                print(f"    Pixels with disturbance: {np.sum(valid):,}")
                print(f"    Min year: {data[valid].min()}")
                print(f"    Max year: {data[valid].max()}")
    else:
        print(f"  WARNING: Template not found: {template_path}")
        print("  Run 02_preprocess_dem.py first, then re-run this script")
        aligned_path = None

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Raw (LANDFIRE grid): {output_path}")
    if aligned_path and aligned_path.exists():
        print(f"  Aligned (project grid): {aligned_path}")
    print("\nNext steps:")
    print("  1. Process Hansen data: python scripts/process_hansen_forest_loss.py")
    print("  2. Delete old forest age: rm data/processed/forest_age.tif")
    print("  3. Rerun vegetation: python scripts/03_preprocess_vegetation.py")


if __name__ == "__main__":
    main()
