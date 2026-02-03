# Washington Cascadia Chanterelle Heat Map

A scientific heat map identifying optimal Pacific Golden Chanterelle (*Cantharellus formosus*) habitat in Washington State's Cascade Range using multi-criteria geospatial analysis.

![Chanterelle Habitat](https://img.shields.io/badge/resolution-30m-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Overview

This project combines multiple environmental datasets to model chanterelle mushroom habitat suitability:

- **Elevation** - Optimal at 2,000-3,000 ft in the Cascades
- **Forest Type** - Douglas-fir, Western Hemlock, and Spruce (mycorrhizal associations)
- **Canopy Cover** - 50-70% (partial shade preferred)
- **Soil pH** - 4.0-5.5 (acidic forest soils)
- **Precipitation** - Higher rainfall areas favored
- **Aspect** - North-facing slopes retain moisture
- **Proximity to Water** - Streams indicate good moisture

## Quick Start

```bash
# Clone the repository
git clone https://github.com/lukaskucinski/wa-chanterelle-finder.git
cd wa-chanterelle-finder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download data (see instructions)
python scripts/01_download_data.py --instructions

# Run preprocessing pipeline
python scripts/02_preprocess_dem.py
python scripts/03_preprocess_vegetation.py
python scripts/04_preprocess_climate.py
python scripts/05_preprocess_soil.py
python scripts/06_preprocess_roads.py

# Calculate suitability
python scripts/07_calculate_suitability.py

# Export for Mapbox
python scripts/08_export_for_mapbox.py
```

## Data Sources

| Dataset | Source | Resolution |
|---------|--------|------------|
| Elevation (DEM) | USGS 3DEP | 10m |
| Vegetation Type | LANDFIRE EVT | 30m |
| Tree Canopy | NLCD 2021 | 30m |
| Precipitation | PRISM | 800m |
| Soil | SSURGO | Variable |
| Roads | OpenStreetMap | Vector |
| Forest Boundaries | USFS | Vector |

## Study Area

Washington State Cascade Range counties:

**West Slope:** Whatcom, Skagit, Snohomish, King, Pierce, Lewis, Cowlitz, Skamania

**East Slope:** Chelan, Kittitas, Yakima

## Output Layers

### 1. Habitat Suitability (0-1)
Pure ecological scoring based on environmental factors. Higher scores indicate better chanterelle habitat.

### 2. Access Quality (0-1)
Road proximity, road type (paved/unpaved), and land ownership (National Forest = legal foraging).

### 3. Combined Score
Habitat × Access for finding accessible high-quality habitat.

## Methodology

### Weighted Overlay Model

| Factor | Weight | Scoring |
|--------|--------|---------|
| Forest Type | 25% | Douglas-fir/Hemlock/Spruce = 1.0 |
| Elevation | 15% | Peak at 2,000-3,000 ft |
| Soil pH | 15% | Optimal 4.0-5.5 |
| Canopy Cover | 10% | Peak at 50-70% |
| Precipitation | 10% | Higher is better |
| Aspect | 10% | North-facing = 1.0 |
| Slope | 5% | Penalty for >30° |
| Water Proximity | 5% | Bonus within 500m |
| Forest Age | 5% | Second-growth optimal |

## Visualization

The heat map uses a chanterelle-themed color palette:

- **Dark Purple** (0.0-0.2): Unsuitable
- **Purple** (0.2-0.4): Poor
- **Orange** (0.4-0.6): Moderate
- **Gold** (0.6-0.8): Good
- **Bright Gold** (0.8-1.0): Excellent

## Project Structure

```
wa_chanterelle_finder/
├── CLAUDE.md                 # Project context
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── data/
│   ├── raw/                  # Downloaded source data
│   ├── processed/            # Intermediate rasters
│   └── output/               # Final suitability maps
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_preprocess_dem.py
│   ├── 03_preprocess_vegetation.py
│   ├── 04_preprocess_climate.py
│   ├── 05_preprocess_soil.py
│   ├── 06_preprocess_roads.py
│   ├── 07_calculate_suitability.py
│   ├── 08_export_for_mapbox.py
│   └── utils/
│       ├── raster_utils.py
│       └── scoring_functions.py
└── mapbox/
    └── style.json            # Mapbox dark theme style
```

## Verification

### Validating the Model

1. **iNaturalist Observations**: Compare against *Cantharellus formosus* records
2. **Known Foraging Areas**: Cross-reference with local mycological society hotspots
3. **Field Validation**: Ground-truth high-scoring areas

## Legal Notice

⚠️ **Foraging Regulations**

- Chanterelle foraging is permitted in National Forests for personal use
- A permit may be required for commercial harvest
- Private land requires permission
- State parks may have restrictions
- Always follow Leave No Trace principles

## References

- Oregon State Extension: *Collecting Wild Mushrooms*
- USFS PNW Research Station: *Chanterelle productivity in young and old forests*
- Washington Department of Natural Resources
- LANDFIRE Program Documentation

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

Areas for improvement:
- Additional validation against known chanterelle sites
- Seasonal moisture modeling
- Climate change projections
- Mobile app integration
