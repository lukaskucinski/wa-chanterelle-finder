"""
Script 06: Preprocess road and access data.

This script processes:
1. OpenStreetMap roads (all road types)
2. Road surface classification (paved/unpaved/trail)
3. Distance to roads raster
4. National Forest boundaries (for legal access)
5. NHD streams (water proximity)

Output:
- Road distance raster
- Road type raster (nearest road type)
- Land ownership raster
- Stream distance raster
"""

import os
import sys
from pathlib import Path
import zipfile

import numpy as np
import geopandas as gpd
from shapely.geometry import box

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
    calculate_distance_raster,
    get_study_area_bounds,
    create_study_area_polygon,
    get_raster_stats,
)

# Directories
RAW_ROADS_DIR = PROJECT_ROOT / "data" / "raw" / "roads"
RAW_BOUNDARIES_DIR = PROJECT_ROOT / "data" / "raw" / "boundaries"
RAW_HYDROLOGY_DIR = PROJECT_ROOT / "data" / "raw" / "hydrology"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# OSM road classification
# highway tag values mapped to road quality
OSM_ROAD_TYPES = {
    # Paved roads (type 1)
    'motorway': 1, 'motorway_link': 1,
    'trunk': 1, 'trunk_link': 1,
    'primary': 1, 'primary_link': 1,
    'secondary': 1, 'secondary_link': 1,
    'tertiary': 1, 'tertiary_link': 1,
    'residential': 1,
    'service': 1,

    # Unpaved/gravel roads (type 2)
    'unclassified': 2,
    'track': 2,

    # Trails/paths (type 3)
    'path': 3,
    'footway': 3,
    'cycleway': 3,
    'bridleway': 3,
}


def extract_osm_data(osm_dir: Path) -> Path:
    """Extract OSM shapefile from zip if needed."""
    zip_files = list(osm_dir.glob("*.zip"))

    for zip_path in zip_files:
        # Check if already extracted
        shp_files = list(osm_dir.glob("*roads*.shp"))
        if shp_files:
            return shp_files[0]

        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(osm_dir)

    # Find extracted shapefile
    shp_files = list(osm_dir.glob("*roads*.shp"))
    if shp_files:
        return shp_files[0]

    return None


def classify_road_surface(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Classify roads by surface type.

    Uses OSM 'highway' and 'surface' tags.
    """
    gdf = gdf.copy()

    # Initialize road type based on highway tag
    if 'fclass' in gdf.columns:
        # Geofabrik format uses 'fclass'
        gdf['road_type'] = gdf['fclass'].map(OSM_ROAD_TYPES).fillna(2).astype(int)
    elif 'highway' in gdf.columns:
        gdf['road_type'] = gdf['highway'].map(OSM_ROAD_TYPES).fillna(2).astype(int)
    else:
        # Default to unpaved
        gdf['road_type'] = 2

    # Refine based on surface tag if available
    if 'surface' in gdf.columns:
        paved_surfaces = ['paved', 'asphalt', 'concrete', 'paving_stones']
        unpaved_surfaces = ['unpaved', 'gravel', 'dirt', 'grass', 'ground', 'compacted']

        for surface in paved_surfaces:
            mask = gdf['surface'].str.lower() == surface.lower()
            gdf.loc[mask, 'road_type'] = 1

        for surface in unpaved_surfaces:
            mask = gdf['surface'].str.lower() == surface.lower()
            gdf.loc[mask & (gdf['road_type'] == 1), 'road_type'] = 2

    return gdf


def process_roads():
    """Main road processing workflow."""
    print("=" * 60)
    print("ROAD & ACCESS PREPROCESSING")
    print("=" * 60)

    # Check for template
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    if not template_path.exists():
        print("ERROR: Template raster not found. Run 02_preprocess_dem.py first.")
        return

    # Output paths
    road_distance_path = PROCESSED_DIR / "road_distance_m.tif"
    road_type_path = PROCESSED_DIR / "road_type.tif"
    ownership_path = PROCESSED_DIR / "land_ownership.tif"
    stream_distance_path = PROCESSED_DIR / "stream_distance_m.tif"

    # Get template info
    template_data, meta = read_raster(template_path)
    valid_mask = ~np.isnan(template_data) & (template_data != NODATA)

    # Get study area
    bounds = get_study_area_bounds()
    study_area = create_study_area_polygon(bounds).to_crs(TARGET_CRS)

    # =========================================================================
    # Process Roads
    # =========================================================================
    print("\n[1/4] Processing road network...")

    osm_dir = RAW_ROADS_DIR / "osm"
    osm_shp = extract_osm_data(osm_dir)

    if osm_shp and osm_shp.exists():
        print(f"Loading roads: {osm_shp}")
        roads_gdf = gpd.read_file(osm_shp)

        # Reproject if needed
        if roads_gdf.crs != TARGET_CRS:
            print("Reprojecting roads to UTM...")
            roads_gdf = roads_gdf.to_crs(TARGET_CRS)

        # Clip to study area (with buffer)
        study_buffered = study_area.buffer(10000)  # 10km buffer
        print("Clipping roads to study area...")
        roads_gdf = gpd.clip(roads_gdf, study_buffered.geometry.values[0])

        print(f"Roads in study area: {len(roads_gdf):,}")

        # Classify road surface
        roads_gdf = classify_road_surface(roads_gdf)

        # Create road distance raster
        if not road_distance_path.exists():
            print("Creating road distance raster...")

            # Rasterize roads (binary)
            roads_binary = rasterize_vector(roads_gdf, template_path, fill_value=0)

            # Calculate distance
            road_distance = calculate_distance_raster(roads_binary, TARGET_RESOLUTION)
            road_distance = np.where(valid_mask, road_distance, NODATA)

            write_raster(road_distance_path, road_distance, meta['transform'], TARGET_CRS)
            print(f"Road distance saved to: {road_distance_path}")

            stats = get_raster_stats(road_distance_path)
            print(f"  Min distance: {stats['min']:.0f}m")
            print(f"  Max distance: {stats['max']:.0f}m")
            print(f"  Mean distance: {stats['mean']:.0f}m")

        # Create road type raster (nearest road type)
        if not road_type_path.exists():
            print("Creating road type raster...")

            # Create separate distance rasters for each road type
            type_distances = {}
            for road_type in [1, 2, 3]:
                type_roads = roads_gdf[roads_gdf['road_type'] == road_type]
                if len(type_roads) > 0:
                    type_binary = rasterize_vector(type_roads, template_path, fill_value=0)
                    type_distances[road_type] = calculate_distance_raster(type_binary, TARGET_RESOLUTION)
                else:
                    type_distances[road_type] = np.full_like(template_data, np.inf)

            # Find nearest road type for each cell
            road_type_raster = np.zeros_like(template_data, dtype=np.uint8)
            min_distance = np.full_like(template_data, np.inf)

            for road_type, distances in type_distances.items():
                closer = distances < min_distance
                road_type_raster[closer] = road_type
                min_distance = np.minimum(min_distance, distances)

            road_type_raster = np.where(valid_mask, road_type_raster, 0)
            write_raster(road_type_path, road_type_raster, meta['transform'], TARGET_CRS, nodata=0, dtype='uint8')
            print(f"Road type saved to: {road_type_path}")

    else:
        print("WARNING: No OSM road data found.")
        print("Creating placeholder road layers...")

        # Placeholder: moderate distance everywhere
        if not road_distance_path.exists():
            road_distance = np.where(valid_mask, 2000, NODATA)  # 2km default
            write_raster(road_distance_path, road_distance, meta['transform'], TARGET_CRS)

        if not road_type_path.exists():
            road_type = np.where(valid_mask, 2, 0).astype(np.uint8)  # Assume unpaved
            write_raster(road_type_path, road_type, meta['transform'], TARGET_CRS, nodata=0, dtype='uint8')

    # =========================================================================
    # Process Land Ownership (National Forest boundaries)
    # =========================================================================
    print("\n[2/4] Processing land ownership...")

    forest_files = list(RAW_BOUNDARIES_DIR.glob("*Forest*.shp")) + list(RAW_BOUNDARIES_DIR.glob("*forest*.shp"))

    if forest_files and not ownership_path.exists():
        print(f"Loading forest boundaries: {forest_files[0]}")
        forests_gdf = gpd.read_file(forest_files[0])

        # Reproject
        if forests_gdf.crs != TARGET_CRS:
            forests_gdf = forests_gdf.to_crs(TARGET_CRS)

        # Filter to Washington forests
        wa_forests = ['Olympic', 'Mt. Baker-Snoqualmie', 'Gifford Pinchot', 'Wenatchee',
                      'Okanogan', 'Colville', 'Mt Baker', 'Snoqualmie']

        if 'FORESTNAME' in forests_gdf.columns:
            wa_mask = forests_gdf['FORESTNAME'].str.contains('|'.join(wa_forests), case=False, na=False)
            forests_gdf = forests_gdf[wa_mask]
        elif 'FORESTORGC' in forests_gdf.columns:
            # WA forest org codes start with '06'
            forests_gdf = forests_gdf[forests_gdf['FORESTORGC'].str.startswith('06', na=False)]

        print(f"Washington forests: {len(forests_gdf)}")

        # Rasterize: 1 = National Forest, 0 = other
        ownership = rasterize_vector(forests_gdf, template_path, fill_value=0)
        ownership = np.where(valid_mask, ownership, 0).astype(np.uint8)

        write_raster(ownership_path, ownership, meta['transform'], TARGET_CRS, nodata=0, dtype='uint8')
        print(f"Land ownership saved to: {ownership_path}")

        nf_pct = np.sum(ownership == 1) / np.sum(valid_mask) * 100
        print(f"  National Forest coverage: {nf_pct:.1f}%")

    elif not ownership_path.exists():
        print("WARNING: No forest boundary data found.")
        print("Creating placeholder (assuming all National Forest)...")

        # Placeholder: all National Forest (conservative for foraging access)
        ownership = np.where(valid_mask, 1, 0).astype(np.uint8)
        write_raster(ownership_path, ownership, meta['transform'], TARGET_CRS, nodata=0, dtype='uint8')

    # =========================================================================
    # Process Streams (water proximity)
    # =========================================================================
    print("\n[3/4] Processing stream network...")

    stream_files = list(RAW_HYDROLOGY_DIR.glob("*Flowline*.shp")) + list(RAW_HYDROLOGY_DIR.glob("*flowline*.shp"))
    stream_files += list(RAW_HYDROLOGY_DIR.glob("*stream*.shp"))

    if stream_files and not stream_distance_path.exists():
        print(f"Loading streams: {stream_files[0]}")
        streams_gdf = gpd.read_file(stream_files[0])

        # Reproject
        if streams_gdf.crs != TARGET_CRS:
            print("Reprojecting streams to UTM...")
            streams_gdf = streams_gdf.to_crs(TARGET_CRS)

        # Clip to study area
        study_buffered = study_area.buffer(10000)
        streams_gdf = gpd.clip(streams_gdf, study_buffered.geometry.values[0])

        print(f"Stream segments in study area: {len(streams_gdf):,}")

        # Rasterize and calculate distance
        streams_binary = rasterize_vector(streams_gdf, template_path, fill_value=0)
        stream_distance = calculate_distance_raster(streams_binary, TARGET_RESOLUTION)
        stream_distance = np.where(valid_mask, stream_distance, NODATA)

        write_raster(stream_distance_path, stream_distance, meta['transform'], TARGET_CRS)
        print(f"Stream distance saved to: {stream_distance_path}")

        stats = get_raster_stats(stream_distance_path)
        print(f"  Min distance: {stats['min']:.0f}m")
        print(f"  Max distance: {stats['max']:.0f}m")
        print(f"  Mean distance: {stats['mean']:.0f}m")

    elif not stream_distance_path.exists():
        print("WARNING: No stream data found.")
        print("Creating placeholder stream distance...")

        # Placeholder: moderate distance to water
        stream_distance = np.where(valid_mask, 500, NODATA)  # 500m default
        write_raster(stream_distance_path, stream_distance, meta['transform'], TARGET_CRS)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n[4/4] Verifying outputs...")

    outputs = [
        ("Road Distance", road_distance_path),
        ("Road Type", road_type_path),
        ("Land Ownership", ownership_path),
        ("Stream Distance", stream_distance_path),
    ]

    for name, path in outputs:
        if path.exists():
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ✗ {name}: NOT CREATED")

    print("\n" + "=" * 60)
    print("Road & access preprocessing complete!")
    print("=" * 60)


def main():
    """Entry point."""
    process_roads()


if __name__ == "__main__":
    main()
