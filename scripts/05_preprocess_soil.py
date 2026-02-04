"""
Script 05: Preprocess soil data from SSURGO/gSSURGO.

This script processes:
1. Soil pH (1:1 water method)
2. Soil drainage class

gSSURGO data structure:
- MapunitRaster_10m: 10m raster with map unit keys (mukey)
- Valu1 table: Pre-summarized attributes including ph1to1h2o_r and drclassdcd

The script joins the Valu1 table to the raster via mukey to create
continuous pH and drainage rasters.

Output:
- Soil pH raster
- Soil drainage class raster
"""

import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

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
GSSURGO_GDB = RAW_SOIL_DIR / "gSSURGO_WA.gdb"
POLARIS_PH_DIR = RAW_SOIL_DIR / "polaris_ph"

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


def read_soil_tables(gdb_path: Path) -> Dict[int, dict]:
    """
    Read soil properties from gSSURGO geodatabase tables.

    Joins component and chorizon tables to get pH and drainage for each mukey.
    SSURGO structure: mukey -> component (cokey) -> chorizon (chkey)

    Returns a dictionary mapping mukey to soil properties.
    """
    try:
        import fiona
        import pandas as pd

        print(f"Reading soil tables from: {gdb_path}")

        # Read component table (has mukey, cokey, drclassdcd, comppct_r)
        print("  Reading component table...")
        with fiona.open(str(gdb_path), layer='component') as src:
            comp_records = [dict(record['properties']) for record in src]
        comp_df = pd.DataFrame(comp_records)
        print(f"    Found {len(comp_df):,} components")

        # Read chorizon table (has cokey, ph1to1h2o_r, hzdept_r, hzdepb_r)
        print("  Reading chorizon table...")
        with fiona.open(str(gdb_path), layer='chorizon') as src:
            hz_records = [dict(record['properties']) for record in src]
        hz_df = pd.DataFrame(hz_records)
        print(f"    Found {len(hz_df):,} horizons")

        # Get drainage from component table (use dominant component per mukey)
        # comppct_r = component percentage, higher = more dominant
        print("  Processing drainage classes...")
        drainage_lookup = {}
        # Column name varies: drclassdcd (older) or drainagecl (newer)
        drain_col = 'drainagecl' if 'drainagecl' in comp_df.columns else 'drclassdcd'
        if drain_col in comp_df.columns:
            # Sort by comppct_r descending to get dominant component first
            comp_sorted = comp_df.sort_values('comppct_r', ascending=False)
            for mukey in comp_sorted['mukey'].unique():
                if pd.isna(mukey):
                    continue
                mukey_comps = comp_sorted[comp_sorted['mukey'] == mukey]
                # Get drainage from dominant component
                for _, row in mukey_comps.iterrows():
                    drain = row.get(drain_col)
                    if drain and not pd.isna(drain):
                        try:
                            drainage_lookup[int(mukey)] = drain
                        except (ValueError, TypeError):
                            pass
                        break

        # Get pH from chorizon table (use surface horizon, weighted by component %)
        print("  Processing soil pH...")
        ph_lookup = {}

        # Build cokey -> mukey mapping
        cokey_to_mukey = {}
        cokey_to_comppct = {}
        for _, row in comp_df.iterrows():
            cokey = row.get('cokey')
            mukey = row.get('mukey')
            comppct = row.get('comppct_r', 0)
            if cokey and mukey:
                try:
                    cokey_to_mukey[int(cokey)] = int(mukey)
                    cokey_to_comppct[int(cokey)] = float(comppct) if comppct else 0
                except (ValueError, TypeError):
                    pass

        # For each horizon, get pH and aggregate by mukey
        # Use surface horizon (hzdept_r closest to 0) weighted by component %
        mukey_ph_data = {}  # mukey -> list of (ph, weight)

        for _, row in hz_df.iterrows():
            cokey = row.get('cokey')
            ph = row.get('ph1to1h2o_r')
            hzdept = row.get('hzdept_r', 0)  # depth to top of horizon

            if cokey is None or ph is None or pd.isna(ph):
                continue

            try:
                cokey = int(cokey)
                mukey = cokey_to_mukey.get(cokey)
                if mukey is None:
                    continue

                # Only use surface horizons (top 30cm)
                if hzdept is not None and hzdept <= 30:
                    comppct = cokey_to_comppct.get(cokey, 1)
                    if mukey not in mukey_ph_data:
                        mukey_ph_data[mukey] = []
                    mukey_ph_data[mukey].append((float(ph), comppct))
            except (ValueError, TypeError):
                continue

        # Calculate weighted average pH per mukey
        for mukey, ph_list in mukey_ph_data.items():
            total_weight = sum(w for _, w in ph_list)
            if total_weight > 0:
                weighted_ph = sum(ph * w for ph, w in ph_list) / total_weight
                ph_lookup[mukey] = weighted_ph

        # Combine into final lookup
        all_mukeys = set(drainage_lookup.keys()) | set(ph_lookup.keys())
        lookup = {}
        for mukey in all_mukeys:
            lookup[mukey] = {
                'ph': ph_lookup.get(mukey),
                'drainage': drainage_lookup.get(mukey),
            }

        # Report stats
        ph_valid = sum(1 for v in lookup.values() if v['ph'] is not None)
        drain_valid = sum(1 for v in lookup.values() if v['drainage'] is not None)
        print(f"  Map units with pH data: {ph_valid:,}")
        print(f"  Map units with drainage data: {drain_valid:,}")

        return lookup

    except ImportError:
        print("ERROR: fiona not installed. Run: pip install fiona")
        return {}
    except Exception as e:
        print(f"ERROR reading soil tables: {e}")
        import traceback
        traceback.print_exc()
        return {}


def process_gssurgo_geodatabase(
    gdb_path: Path,
    template_path: Path
) -> tuple:
    """
    Process gSSURGO File Geodatabase to extract soil pH and drainage rasters.

    Steps:
    1. Read component/chorizon tables for soil properties
    2. Rasterize MUPOLYGON layer with mukey values
    3. Join to create pH and drainage rasters
    4. Reproject/resample to match template
    """
    print("\n" + "=" * 60)
    print("Processing gSSURGO Geodatabase")
    print("=" * 60)

    # Read soil properties from component/chorizon tables
    lookup = read_soil_tables(gdb_path)
    if not lookup:
        return None, None

    # Read MUPOLYGON and rasterize
    print("\nReading MUPOLYGON layer...")
    try:
        gdf = gpd.read_file(str(gdb_path), layer='MUPOLYGON')
        print(f"  Found {len(gdf):,} polygons")

        # Ensure MUKEY is numeric
        gdf['MUKEY'] = pd.to_numeric(gdf['MUKEY'], errors='coerce')

        # Create pH and drainage columns by joining lookup
        print("  Joining soil properties...")
        gdf['ph_value'] = gdf['MUKEY'].map(lambda x: lookup.get(int(x), {}).get('ph') if pd.notna(x) else None)
        gdf['drain_value'] = gdf['MUKEY'].map(
            lambda x: DRAINAGE_CODES.get(lookup.get(int(x), {}).get('drainage'), 0) if pd.notna(x) else 0
        )

        # Report coverage
        ph_coverage = gdf['ph_value'].notna().sum() / len(gdf) * 100
        drain_coverage = (gdf['drain_value'] > 0).sum() / len(gdf) * 100
        print(f"  pH coverage: {ph_coverage:.1f}% of polygons")
        print(f"  Drainage coverage: {drain_coverage:.1f}% of polygons")

    except Exception as e:
        print(f"ERROR reading MUPOLYGON: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Reproject to match template CRS
    print("\nReprojecting to template CRS...")
    with rasterio.open(template_path) as template:
        dst_crs = template.crs
        dst_transform = template.transform
        dst_shape = (template.height, template.width)

    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
        print(f"  Reprojected to {dst_crs}")

    # Rasterize pH
    print("\nRasterizing soil pH...")
    from rasterio.features import rasterize

    # Filter to polygons with pH data
    gdf_ph = gdf[gdf['ph_value'].notna()].copy()
    if len(gdf_ph) > 0:
        shapes_ph = ((geom, val) for geom, val in zip(gdf_ph.geometry, gdf_ph['ph_value']))
        ph_raster = rasterize(
            shapes_ph,
            out_shape=dst_shape,
            transform=dst_transform,
            fill=NODATA,
            dtype=np.float32
        )
        print(f"  Rasterized {len(gdf_ph):,} polygons with pH data")
    else:
        ph_raster = np.full(dst_shape, NODATA, dtype=np.float32)
        print("  WARNING: No pH data to rasterize")

    # Rasterize drainage
    print("Rasterizing drainage class...")
    gdf_drain = gdf[gdf['drain_value'] > 0].copy()
    if len(gdf_drain) > 0:
        shapes_drain = ((geom, val) for geom, val in zip(gdf_drain.geometry, gdf_drain['drain_value']))
        drainage_raster = rasterize(
            shapes_drain,
            out_shape=dst_shape,
            transform=dst_transform,
            fill=0,
            dtype=np.float32
        )
        print(f"  Rasterized {len(gdf_drain):,} polygons with drainage data")
    else:
        drainage_raster = np.full(dst_shape, 0, dtype=np.float32)
        print("  WARNING: No drainage data to rasterize")

    # Report final coverage
    ph_valid = np.sum(ph_raster != NODATA)
    drain_valid = np.sum(drainage_raster > 0)
    total = dst_shape[0] * dst_shape[1]
    print(f"\nFinal raster coverage:")
    print(f"  pH: {ph_valid:,} / {total:,} pixels ({100*ph_valid/total:.1f}%)")
    print(f"  Drainage: {drain_valid:,} / {total:,} pixels ({100*drain_valid/total:.1f}%)")

    return ph_raster, drainage_raster


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


def process_polaris_ph(template_path: Path) -> Optional[np.ndarray]:
    """
    Process POLARIS soil pH data.

    POLARIS provides 30m pH data derived from SSURGO.
    This function merges tiles, clips to study area, and aligns to template.

    Returns pH raster array or None if no data found.
    """
    from rasterio.merge import merge

    if not POLARIS_PH_DIR.exists():
        print("  POLARIS pH directory not found")
        return None

    # Find all pH tiles
    ph_tiles = list(POLARIS_PH_DIR.glob("*.tif"))
    if not ph_tiles:
        print("  No POLARIS pH tiles found")
        return None

    print(f"\nProcessing POLARIS pH data...")
    print(f"  Found {len(ph_tiles)} tiles")

    # Merge tiles
    print("  Merging tiles...")
    src_files = [rasterio.open(f) for f in ph_tiles]

    try:
        mosaic, mosaic_transform = merge(src_files)
        mosaic = mosaic[0]  # Get first band

        # Get CRS from first file
        src_crs = src_files[0].crs
        src_nodata = src_files[0].nodata

        print(f"  Merged shape: {mosaic.shape}")
        print(f"  Source CRS: {src_crs}")

    finally:
        for src in src_files:
            src.close()

    # Handle nodata - POLARIS uses -9999 or similar
    if src_nodata is not None:
        mosaic = np.where(mosaic == src_nodata, np.nan, mosaic)

    # POLARIS pH values are multiplied by 10 (e.g., 55 = pH 5.5)
    # Check if values need scaling
    valid_vals = mosaic[~np.isnan(mosaic)]
    if len(valid_vals) > 0 and valid_vals.mean() > 14:  # pH can't be > 14
        print("  Scaling pH values (dividing by 10)...")
        mosaic = mosaic / 10.0

    # Reproject to match template
    print("  Reprojecting to match template...")
    with rasterio.open(template_path) as template:
        dst_crs = template.crs
        dst_transform = template.transform
        dst_shape = (template.height, template.width)

    # Reproject
    ph_reprojected = np.full(dst_shape, NODATA, dtype=np.float32)
    reproject(
        source=mosaic.astype(np.float32),
        destination=ph_reprojected,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=NODATA
    )

    # Report coverage
    valid_count = np.sum(ph_reprojected != NODATA)
    total_count = ph_reprojected.size
    print(f"  pH coverage: {valid_count:,} / {total_count:,} pixels ({100*valid_count/total_count:.1f}%)")

    if valid_count == 0:
        return None

    # Report stats
    valid_vals = ph_reprojected[ph_reprojected != NODATA]
    print(f"  pH range: {valid_vals.min():.2f} - {valid_vals.max():.2f}")
    print(f"  pH mean: {valid_vals.mean():.2f}")

    return ph_reprojected


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

    # Try different data sources in order of preference

    # 1. Try gSSURGO geodatabase (best option - 10m resolution)
    if GSSURGO_GDB.exists() and (ph_raster is None or drainage_raster is None):
        print("\n[1/2] Processing gSSURGO geodatabase...")
        try:
            ph_raster, drainage_raster = process_gssurgo_geodatabase(GSSURGO_GDB, template_path)
        except Exception as e:
            print(f"Error processing gSSURGO: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to other methods...")

    # 2. Try standalone gSSURGO raster (if extracted separately)
    if 'gssurgo_raster' in soil_files and (ph_raster is None or drainage_raster is None):
        print("\n[1/2] Found gSSURGO raster file...")
        print("Note: Standalone raster requires separate Valu1 table for attributes")
        print("Prefer using the full geodatabase instead")

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

    # Check if pH has valid data (not just all NODATA)
    ph_has_data = ph_raster is not None and np.sum(ph_raster != NODATA) > 0
    drain_has_data = drainage_raster is not None and np.sum(drainage_raster > 0) > 0

    # Try POLARIS for pH if gSSURGO didn't provide it
    if not ph_has_data and POLARIS_PH_DIR.exists():
        print("\nTrying POLARIS for pH data...")
        polaris_ph = process_polaris_ph(template_path)
        if polaris_ph is not None:
            ph_raster = polaris_ph
            ph_has_data = True

    # Fall back to defaults if still needed
    if not ph_has_data or not drain_has_data:
        print("\nSome soil data missing. Using defaults where needed.")
        ph_default, drainage_default = create_default_soil_layers(template_path)

        if not ph_has_data:
            print("  - Using default pH values (typical Cascades acidic soils)")
            ph_raster = ph_default
        if not drain_has_data:
            print("  - Using default drainage values")
            drainage_raster = drainage_default

    # Get metadata from template
    _, meta = read_raster(template_path)

    # Save pH raster
    print("\n[2/2] Saving soil layers...")

    if not ph_path.exists():
        write_raster(ph_path, ph_raster, meta['transform'], TARGET_CRS)
        print(f"Soil pH saved to: {ph_path}")

        stats = get_raster_stats(ph_path)
        if stats and stats.get('min') is not None:
            print(f"\nSoil pH Statistics:")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  Mean: {stats['mean']:.2f}")
        else:
            print("\nSoil pH Statistics: No valid data")

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
