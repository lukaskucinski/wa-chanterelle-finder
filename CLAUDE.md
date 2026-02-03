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

- **DEM**: USGS 3DEP (10m)
- **Vegetation**: LANDFIRE EVT (30m)
- **Canopy**: NLCD Tree Canopy Cover 2021 (30m)
- **Climate**: PRISM (800m)
- **Soil**: SSURGO/gSSURGO
- **Roads**: OpenStreetMap + WSDOT

## Study Area

Cascade Range counties:
- West Slope: Whatcom, Skagit, Snohomish, King, Pierce, Lewis, Cowlitz, Skamania
- East Slope: Chelan, Kittitas, Yakima

## Running the Pipeline

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

## Output Layers

1. **Habitat Suitability** - Pure ecological scoring (0-1)
2. **Access Quality** - Road proximity and quality (0-1)
3. **Reference Layers** - National Forest boundaries, roads by type
