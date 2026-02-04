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
