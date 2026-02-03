"""
Script 01: Download source data for chanterelle habitat analysis.

Downloads data from:
- USGS 3DEP (DEM)
- LANDFIRE (vegetation)
- NLCD (canopy cover)
- PRISM (climate)
- OpenStreetMap (roads)
- USFS (forest boundaries)

Many datasets require manual download due to size or authentication.
This script provides download URLs and instructions.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.raster_utils import get_study_area_bounds

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def print_download_instructions():
    """Print manual download instructions for large datasets."""
    bounds = get_study_area_bounds()

    instructions = f"""
================================================================================
CHANTERELLE HABITAT DATA DOWNLOAD INSTRUCTIONS
================================================================================

Study Area Bounds (WGS84):
  West:  {bounds['west']}
  East:  {bounds['east']}
  South: {bounds['south']}
  North: {bounds['north']}

--------------------------------------------------------------------------------
1. DIGITAL ELEVATION MODEL (DEM) - USGS 3DEP
--------------------------------------------------------------------------------
Source: https://apps.nationalmap.gov/downloader/

Steps:
1. Go to the National Map Downloader
2. Draw a box covering Washington Cascades (approx {bounds['west']} to {bounds['east']})
3. Select "Elevation Products (3DEP)" → "1/3 arc-second DEM"
4. Download all tiles covering the study area
5. Save to: {RAW_DATA_DIR / 'dem'}

Expected files: Multiple .tif files (e.g., USGS_13_n47w122_20230601.tif)

--------------------------------------------------------------------------------
2. LANDFIRE EXISTING VEGETATION TYPE (EVT)
--------------------------------------------------------------------------------
Source: https://landfire.gov/evt.php

Steps:
1. Go to LANDFIRE Data Distribution Site
2. Select "LF 2022" → "Existing Vegetation Type"
3. Download for CONUS or clip to Pacific Northwest
4. Save to: {RAW_DATA_DIR / 'vegetation'}

Expected file: LF2022_EVT_220_CONUS.tif (or regional clip)

--------------------------------------------------------------------------------
3. NLCD TREE CANOPY COVER 2021
--------------------------------------------------------------------------------
Source: https://www.mrlc.gov/data/nlcd-2021-usfs-tree-canopy-cover-conus

Steps:
1. Go to MRLC Data page
2. Download "NLCD 2021 USFS Tree Canopy Cover (CONUS)"
3. Save to: {RAW_DATA_DIR / 'canopy'}

Expected file: nlcd_tcc_conus_2021_v2021-4.tif

--------------------------------------------------------------------------------
4. PRISM CLIMATE DATA (Precipitation)
--------------------------------------------------------------------------------
Source: https://prism.oregonstate.edu/normals/

Steps:
1. Go to PRISM Climate Group
2. Download 30-year normals (1991-2020):
   - Annual precipitation (ppt)
   - Fall precipitation (Sep, Oct, Nov monthly normals)
3. Save to: {RAW_DATA_DIR / 'climate'}

Expected files:
- PRISM_ppt_30yr_normal_800mM4_annual_bil.bil
- PRISM_ppt_30yr_normal_800mM4_09_bil.bil (September)
- PRISM_ppt_30yr_normal_800mM4_10_bil.bil (October)
- PRISM_ppt_30yr_normal_800mM4_11_bil.bil (November)

--------------------------------------------------------------------------------
5. SSURGO SOIL DATA
--------------------------------------------------------------------------------
Source: https://websoilsurvey.nrcs.usda.gov/

Steps:
1. Go to Web Soil Survey
2. Define Area of Interest covering WA Cascades
3. Download soil data including:
   - pH (1:1 water)
   - Drainage class
4. Export as shapefile or geodatabase
5. Save to: {RAW_DATA_DIR / 'soil'}

Alternative: Use gSSURGO from NRCS Geospatial Data Gateway
https://datagateway.nrcs.usda.gov/

--------------------------------------------------------------------------------
6. OPENSTREETMAP ROADS (Washington)
--------------------------------------------------------------------------------
Source: https://download.geofabrik.de/north-america/us/washington.html

Steps:
1. Download washington-latest-free.shp.zip
2. Extract to: {RAW_DATA_DIR / 'roads' / 'osm'}

Expected files: gis_osm_roads_free_1.shp (and associated files)

--------------------------------------------------------------------------------
7. USFS ADMINISTRATIVE BOUNDARIES
--------------------------------------------------------------------------------
Source: https://data.fs.usda.gov/geodata/edw/datasets.php?dsetCategory=boundaries

Steps:
1. Download "Administrative Forest Boundaries"
2. Save to: {RAW_DATA_DIR / 'boundaries'}

Expected file: S_USA.AdministrativeForest.shp

--------------------------------------------------------------------------------
8. NHD HYDROGRAPHY (Streams/Rivers)
--------------------------------------------------------------------------------
Source: https://www.usgs.gov/national-hydrography/access-national-hydrography-products

Steps:
1. Go to National Hydrography Dataset
2. Download NHD High Resolution for Washington (HU4 watersheds 1707, 1708, 1709)
3. Save to: {RAW_DATA_DIR / 'hydrology'}

Expected files: NHDFlowline.shp, NHDWaterbody.shp

--------------------------------------------------------------------------------
DATA DIRECTORY STRUCTURE
--------------------------------------------------------------------------------
After downloading, your data/raw folder should look like:

{RAW_DATA_DIR}/
├── dem/
│   └── *.tif (DEM tiles)
├── vegetation/
│   └── LF2022_EVT_*.tif
├── canopy/
│   └── nlcd_tcc_*.tif
├── climate/
│   └── PRISM_ppt_*.bil
├── soil/
│   └── gSSURGO_WA.gdb/ or shapefiles
├── roads/
│   └── osm/
│       └── gis_osm_roads_free_1.shp
├── boundaries/
│   └── S_USA.AdministrativeForest.shp
└── hydrology/
    └── NHDFlowline.shp

================================================================================
"""
    print(instructions)


def download_osm_washington():
    """Download OpenStreetMap data for Washington State."""
    osm_dir = RAW_DATA_DIR / "roads" / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)

    url = "https://download.geofabrik.de/north-america/us/washington-latest-free.shp.zip"
    dest = osm_dir / "washington-latest-free.shp.zip"

    if dest.exists():
        print(f"OSM data already exists: {dest}")
        return True

    print("Downloading OpenStreetMap Washington data...")
    return download_file(url, dest)


def create_directory_structure():
    """Create the required directory structure for raw data."""
    subdirs = [
        "dem",
        "vegetation",
        "canopy",
        "climate",
        "soil",
        "roads/osm",
        "boundaries",
        "hydrology",
    ]

    for subdir in subdirs:
        path = RAW_DATA_DIR / subdir
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")


def check_data_availability():
    """Check which data files are already present."""
    checks = {
        "DEM": RAW_DATA_DIR / "dem",
        "Vegetation (LANDFIRE)": RAW_DATA_DIR / "vegetation",
        "Canopy Cover (NLCD)": RAW_DATA_DIR / "canopy",
        "Climate (PRISM)": RAW_DATA_DIR / "climate",
        "Soil (SSURGO)": RAW_DATA_DIR / "soil",
        "Roads (OSM)": RAW_DATA_DIR / "roads" / "osm",
        "Forest Boundaries": RAW_DATA_DIR / "boundaries",
        "Hydrology (NHD)": RAW_DATA_DIR / "hydrology",
    }

    print("\n" + "=" * 60)
    print("DATA AVAILABILITY CHECK")
    print("=" * 60)

    for name, path in checks.items():
        if path.exists():
            files = list(path.glob("*"))
            # Filter out directories for count
            files = [f for f in files if f.is_file()]
            if files:
                print(f"✓ {name}: {len(files)} file(s) found")
            else:
                print(f"✗ {name}: Directory exists but empty")
        else:
            print(f"✗ {name}: Not found")

    print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download data for chanterelle habitat analysis")
    parser.add_argument("--setup", action="store_true", help="Create directory structure")
    parser.add_argument("--check", action="store_true", help="Check data availability")
    parser.add_argument("--osm", action="store_true", help="Download OSM data")
    parser.add_argument("--instructions", action="store_true", help="Print download instructions")

    args = parser.parse_args()

    if args.setup or not any([args.check, args.osm, args.instructions]):
        create_directory_structure()

    if args.check:
        check_data_availability()

    if args.osm:
        download_osm_washington()

    if args.instructions or not any([args.setup, args.check, args.osm]):
        print_download_instructions()


if __name__ == "__main__":
    main()
