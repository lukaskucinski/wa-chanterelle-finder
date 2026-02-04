# Washington Cascadia Chanterelle Heat Map

## Project Guidelines

- **No AI attribution**: Never include co-authorship credits, signatures, or attribution to Claude, Anthropic, or any AI assistant in commits, code comments, or documentation.

## Project Overview

This project creates a scientific heat map identifying optimal chanterelle mushroom (Cantharellus formosus) habitat in Washington State's Cascade Range using multi-criteria geospatial analysis.

## Key Technologies

- **Python**: rasterio, geopandas, numpy, scipy for geospatial processing
- **Visualization**: Mapbox (dark theme) with Cloud-Optimized GeoTIFF
- **Resolution**: 30 meters (EPSG:32610 - UTM 10N)

## Chanterelle Habitat Factors

| Factor | Optimal Conditions | Weight |
|--------|-------------------|--------|
| Elevation | 1,500-4,000 ft (peak 2,000-3,000 ft) | 15% |
| Forest Type | Douglas-fir, Western Hemlock, Spruce | 25% |
| Canopy Cover | 40-80% (peak 50-70%) | 10% |
| Soil pH | 4.0-5.5 (acidic) | 15% |
| Precipitation | Higher is better (up to ~80"/yr) | 10% |
| Slope | Penalty for >30 degrees | 5% |
| Aspect | North-facing preferred (315-45°) | 10% |
| Water Proximity | Bonus within 500m of streams | 5% |
| Forest Age | Second-growth (40-80 yr) optimal | 5% |

## Directory Structure

```
data/raw/           # Downloaded source data
data/processed/     # Intermediate processed rasters
data/output/        # Final heat map outputs
scripts/            # Processing scripts (01-08)
scripts/utils/      # Utility modules
notebooks/          # Exploration notebooks
mapbox/             # Mapbox style configuration
```

## Data Sources

| Layer | Source | Resolution | Notes |
|-------|--------|------------|-------|
| DEM | USGS 3DEP via AWS S3 | 10m | Tiles merged with BIGTIFF support |
| Vegetation | LANDFIRE EVT 2024 | 30m | LC24_EVT_250 |
| Canopy | NLCD Tree Canopy Cover 2023 | 30m | Values >100 masked as nodata |
| Climate | PRISM 30-year normals | 800m | Annual + Sep/Oct/Nov/Dec for fall precip |
| Soil pH | POLARIS | 30m | 32 tiles covering WA (mean pH 0-5cm depth) |
| Soil Drainage | gSSURGO | 30m | Rasterized from MUPOLYGON layer |
| Roads | OpenStreetMap | Vector | Geofabrik WA extract |
| Streams | OpenStreetMap | Vector | Waterways from Geofabrik |
| Forest Disturbance | Hansen GFC + LANDFIRE | 30m | Combined dataset (see below) |

### Data Source URLs

| Dataset | URL |
|---------|-----|
| USGS 3DEP | `s3://prd-tnm/StagedProducts/Elevation/13/TIFF/current/` |
| LANDFIRE EVT | https://landfire.gov/viewer/ (manual download) |
| LANDFIRE Disturbance | https://landfire.gov/version_download.php |
| NLCD Tree Canopy | https://www.mrlc.gov/data |
| PRISM Climate | https://prism.oregonstate.edu/normals/ |
| POLARIS Soil | http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/ph/mean/0_5/ |
| gSSURGO | https://nrcs.app.box.com/v/soils (WA state geodatabase) |
| Hansen GFC | https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/ |
| OSM Roads | https://download.geofabrik.de/north-america/us/washington.html |

### Forest Age / Disturbance Data

Forest age is derived from two complementary disturbance datasets:

1. **Hansen Global Forest Change v1.12** (2001-2024)
   - Source: University of Maryland / Google Earth Engine
   - URL: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/
   - Captures: All forest loss (logging, fire, disease, development)
   - Tiles: 50N_130W, 50N_120W

2. **LANDFIRE Annual Disturbance** (1999-2024)
   - Source: LANDFIRE (landfire.gov)
   - 26 annual layers combined into single "last disturbance year" raster
   - Captures: Fire, harvest, insects, disease

**Combined approach**: For each pixel, the more recent disturbance year from either dataset is used. This provides ~60% more coverage than either dataset alone:
- Hansen only: 9.4M pixels
- LANDFIRE only: 11.0M pixels
- Combined: 15.1M pixels

**Limitation**: Data only spans 1999-2024, so we can only identify forests 0-25 years old. Older forests (including optimal second-growth 40-80 years) are assigned a default age of 100 years.

## Study Area

Cascade Range counties:
- West Slope: Whatcom, Skagit, Snohomish, King, Pierce, Lewis, Cowlitz, Skamania
- East Slope: Chelan, Kittitas, Yakima

## Running the Pipeline

### Main Pipeline
```bash
python scripts/01_download_data.py
python scripts/02_preprocess_dem.py
python scripts/03_preprocess_vegetation.py
python scripts/04_preprocess_climate.py
python scripts/05_preprocess_soil.py
python scripts/06_preprocess_roads.py
python scripts/07_calculate_suitability.py
python scripts/08_export_for_mapbox.py
```

### Supplementary Data Scripts
```bash
# Soil pH from POLARIS (run before 05_preprocess_soil.py)
python scripts/download_polaris_ph.py

# Forest disturbance data (run before 03_preprocess_vegetation.py)
python scripts/download_hansen_forest_loss.py
python scripts/process_hansen_forest_loss.py
python scripts/process_annual_disturbance.py
```

## Processed Outputs

| File | Description |
|------|-------------|
| `elevation_ft.tif` | Elevation in feet (template raster) |
| `slope_degrees.tif` | Slope in degrees |
| `aspect.tif` | Aspect in degrees (0-360) |
| `forest_type.tif` | Classified (0=non-forest, 1=other, 2=marginal, 3=optimal) |
| `canopy_cover.tif` | Tree canopy percentage (0-100) |
| `forest_age.tif` | Years since disturbance (0-100) |
| `annual_precip_inches.tif` | Annual precipitation |
| `fall_precip_inches.tif` | Sep+Oct+Nov+Dec precipitation |
| `soil_ph.tif` | Soil pH (0-14 scale) |
| `soil_drainage.tif` | Drainage class (1-7) |
| `road_distance_m.tif` | Distance to nearest road |
| `stream_distance_m.tif` | Distance to nearest stream |
| `land_ownership.tif` | National Forest boundaries |

## Output Layers

1. **Habitat Suitability** - Pure ecological scoring (0-1)
2. **Access Quality** - Road proximity and quality (0-1)
3. **Reference Layers** - National Forest boundaries, roads by type

## Mapbox Deployment

### Setup (One Time)
```bash
# Install dependencies
pip install boto3 python-dotenv

# Create credentials file
copy .env.example .env
# Edit .env with your MAPBOX_ACCESS_TOKEN and MAPBOX_USERNAME
```

### Upload Workflow
```bash
# Export 8-bit COGs (Mapbox requirement)
python scripts/08_export_for_mapbox.py

# Upload to Mapbox
python scripts/upload_to_mapbox.py           # Upload all
python scripts/upload_to_mapbox.py habitat   # Upload habitat only
python scripts/upload_to_mapbox.py access    # Upload access only
```

### COG Format
Mapbox requires 8-bit TIFFs. The export script converts float scores (0-1) to uint8:
- Value 0 = nodata (transparent)
- Value 1-255 = suitability score (1=0%, 255=100%)
- To convert back: `score = (pixel_value - 1) / 254`

### Mapbox Studio
1. Create new style from "Dark" template
2. Add raster layers from uploaded tilesets
3. Configure opacity and blending
4. Publish and share

### Tileset IDs
- `{username}.chanterelle-habitat` - Habitat suitability
- `{username}.chanterelle-access` - Access quality

## Data Processing Notes

### Raw Data Cleanup

After processing, the following raw data was deleted to save disk space (~31 GB):

| Deleted | Size | Reason |
|---------|------|--------|
| `data/raw/dem/*.tif` | 8.3 GB | 20 DEM tiles merged into processed output |
| `data/raw/vegetation/landfire_hdist/extracted/` | 9.6 GB | Extracted annual TIFs no longer needed |
| `data/raw/forest_loss/*treecover2000*` | 708 MB | Optional Hansen data not used |
| `data/processed/dem_merged.tif` | 8.8 GB | Intermediate file |
| `data/processed/dem_clipped.tif` | 3.4 GB | Intermediate file |

**Retained raw data** (needed for potential reprocessing):
- `vegetation/landfire_hdist/*.zip` - Original LANDFIRE downloads
- `canopy/*.tif` - NLCD source (3.5 GB)
- `soil/gSSURGO_WA.gdb/` - Soil geodatabase (2.3 GB)
- `climate/` - PRISM normals (293 MB)

See `data/raw/dem/README.md` for DEM re-download instructions.

### Processing Workflow

1. **DEM Processing** (`02_preprocess_dem.py`)
   - Downloads 20 tiles from AWS S3 covering 45°N-49°N, 120°W-123°W
   - Merges tiles using BIGTIFF for files >4GB
   - Reprojects to UTM 10N (EPSG:32610)
   - Derives slope and aspect

2. **Vegetation Processing** (`03_preprocess_vegetation.py`)
   - Classifies LANDFIRE EVT codes into chanterelle habitat quality
   - Processes NLCD canopy cover (masks nodata values >100)
   - Calculates forest age from combined disturbance data

3. **Climate Processing** (`04_preprocess_climate.py`)
   - Reprojects 800m PRISM data to 30m grid
   - Calculates fall precipitation (Sep+Oct+Nov+Dec)

4. **Soil Processing** (`05_preprocess_soil.py`)
   - Merges 32 POLARIS pH tiles for Washington
   - Extracts drainage class from gSSURGO geodatabase
   - Rasterizes MUPOLYGON layer for drainage

5. **Roads/Access Processing** (`06_preprocess_roads.py`)
   - Calculates distance to nearest road
   - Classifies road types (paved/unpaved/trail)
   - Identifies National Forest boundaries
   - Calculates distance to streams

6. **Suitability Calculation** (`07_calculate_suitability.py`)
   - Applies scoring functions to each factor
   - Combines using weighted overlay
   - Outputs habitat suitability and access quality scores

7. **Export for Mapbox** (`08_export_for_mapbox.py`)
   - Converts float32 scores to 8-bit (Mapbox requirement)
   - Creates Cloud-Optimized GeoTIFFs with overviews
   - Generates preview PNGs
