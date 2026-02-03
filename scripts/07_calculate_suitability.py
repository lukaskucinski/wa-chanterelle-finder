"""
Script 07: Calculate habitat suitability using weighted overlay.

This script:
1. Loads all preprocessed rasters
2. Applies scoring functions to each layer
3. Performs weighted overlay for habitat suitability
4. Creates separate access quality layer
5. Outputs final suitability rasters

Habitat Factors and Weights:
- Elevation (15%): Peak at 2,000-3,000 ft
- Forest Type (25%): Douglas-fir, Hemlock, Spruce
- Canopy Cover (10%): 50-70% optimal
- Soil pH (15%): 4.0-5.5 acidic
- Precipitation (10%): Higher is better
- Slope (5%): Penalty for steep terrain
- Aspect (10%): North-facing preferred
- Water Proximity (5%): Bonus near streams
- Forest Age (5%): Second-growth optimal

Access Factors:
- Road Distance (60%): Closer is better
- Road Type (30%): Paved > Unpaved > Trail
- Land Access (10%): National Forest = legal foraging
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.raster_utils import (
    TARGET_CRS,
    NODATA,
    read_raster,
    write_raster,
    get_raster_stats,
)

from scripts.utils.scoring_functions import (
    score_elevation,
    score_canopy_cover,
    score_soil_ph,
    score_precipitation,
    score_slope,
    score_aspect,
    score_water_proximity,
    score_forest_age,
    score_road_distance,
    score_road_type,
    score_land_ownership,
    weighted_overlay,
)

# Directories
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Habitat suitability weights (must sum to 1.0)
HABITAT_WEIGHTS = {
    'elevation': 0.15,
    'forest_type': 0.25,
    'canopy_cover': 0.10,
    'soil_ph': 0.15,
    'precipitation': 0.10,
    'slope': 0.05,
    'aspect': 0.10,
    'water_proximity': 0.05,
    'forest_age': 0.05,
}

# Access quality weights (must sum to 1.0)
ACCESS_WEIGHTS = {
    'road_distance': 0.60,
    'road_type': 0.30,
    'land_ownership': 0.10,
}


def load_layer(name: str, required: bool = True) -> tuple:
    """
    Load a preprocessed raster layer.

    Returns (data, valid_mask) tuple.
    """
    # Map layer names to file paths
    file_map = {
        'elevation': 'elevation_ft.tif',
        'slope': 'slope_degrees.tif',
        'aspect': 'aspect_degrees.tif',
        'forest_type': 'forest_type.tif',
        'canopy_cover': 'canopy_cover.tif',
        'forest_age': 'forest_age.tif',
        'soil_ph': 'soil_ph.tif',
        'soil_drainage': 'soil_drainage.tif',
        'precipitation': 'annual_precip_inches.tif',
        'road_distance': 'road_distance_m.tif',
        'road_type': 'road_type.tif',
        'land_ownership': 'land_ownership.tif',
        'water_proximity': 'stream_distance_m.tif',
    }

    filepath = PROCESSED_DIR / file_map.get(name, f"{name}.tif")

    if not filepath.exists():
        if required:
            raise FileNotFoundError(f"Required layer not found: {filepath}")
        else:
            print(f"  WARNING: Optional layer not found: {name}")
            return None, None

    data, meta = read_raster(filepath)

    # Create valid mask
    nodata = meta.get('nodata', NODATA)
    valid_mask = ~np.isnan(data)
    if nodata is not None:
        valid_mask &= (data != nodata)

    # Set invalid values to NaN for processing
    data = np.where(valid_mask, data, np.nan)

    return data, valid_mask


def score_forest_type_layer(forest_type: np.ndarray) -> np.ndarray:
    """
    Score forest type classification.

    Input codes:
    - 3: Optimal (Douglas-fir, Hemlock, Spruce)
    - 2: Marginal (Other conifers)
    - 1: Other forest
    - 0: Non-forest
    """
    scores = np.zeros_like(forest_type, dtype=np.float32)

    scores[forest_type == 3] = 1.0   # Optimal chanterelle forests
    scores[forest_type == 2] = 0.6   # Marginal - some potential
    scores[forest_type == 1] = 0.3   # Other forest - low potential
    scores[forest_type == 0] = 0.0   # Non-forest - no chanterelles

    return scores


def calculate_habitat_suitability():
    """Calculate habitat suitability score."""
    print("\n" + "-" * 40)
    print("CALCULATING HABITAT SUITABILITY")
    print("-" * 40)

    # Load all layers
    print("\nLoading layers...")
    layers = {}
    valid_masks = []

    # Required layers
    for name in ['elevation', 'slope', 'aspect']:
        data, mask = load_layer(name, required=True)
        layers[name] = data
        valid_masks.append(mask)
        print(f"  ✓ {name}")

    # Optional layers (use placeholders if missing)
    for name in ['forest_type', 'canopy_cover', 'forest_age', 'soil_ph', 'precipitation', 'water_proximity']:
        data, mask = load_layer(name, required=False)
        if data is not None:
            layers[name] = data
            valid_masks.append(mask)
            print(f"  ✓ {name}")
        else:
            print(f"  - {name} (will use default)")

    # Combined valid mask
    combined_valid = valid_masks[0].copy()
    for mask in valid_masks[1:]:
        combined_valid &= mask

    print(f"\nValid cells: {np.sum(combined_valid):,}")

    # Apply scoring functions
    print("\nScoring layers...")
    scored_layers = []
    weights = []

    # Elevation
    if 'elevation' in layers:
        scored = score_elevation(layers['elevation'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['elevation'])
        print(f"  Elevation: mean score = {np.nanmean(scored[combined_valid]):.3f}")

    # Forest Type
    if 'forest_type' in layers:
        scored = score_forest_type_layer(layers['forest_type'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['forest_type'])
        print(f"  Forest Type: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        # Default: assume good forest
        scored_layers.append(np.where(combined_valid, 0.8, 0))
        weights.append(HABITAT_WEIGHTS['forest_type'])
        print(f"  Forest Type: using default (0.8)")

    # Canopy Cover
    if 'canopy_cover' in layers:
        scored = score_canopy_cover(layers['canopy_cover'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['canopy_cover'])
        print(f"  Canopy Cover: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.7, 0))
        weights.append(HABITAT_WEIGHTS['canopy_cover'])
        print(f"  Canopy Cover: using default (0.7)")

    # Soil pH
    if 'soil_ph' in layers:
        scored = score_soil_ph(layers['soil_ph'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['soil_ph'])
        print(f"  Soil pH: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.7, 0))
        weights.append(HABITAT_WEIGHTS['soil_ph'])
        print(f"  Soil pH: using default (0.7)")

    # Precipitation
    if 'precipitation' in layers:
        scored = score_precipitation(layers['precipitation'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['precipitation'])
        print(f"  Precipitation: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.8, 0))
        weights.append(HABITAT_WEIGHTS['precipitation'])
        print(f"  Precipitation: using default (0.8)")

    # Slope
    if 'slope' in layers:
        scored = score_slope(layers['slope'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['slope'])
        print(f"  Slope: mean score = {np.nanmean(scored[combined_valid]):.3f}")

    # Aspect
    if 'aspect' in layers:
        scored = score_aspect(layers['aspect'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['aspect'])
        print(f"  Aspect: mean score = {np.nanmean(scored[combined_valid]):.3f}")

    # Water Proximity
    if 'water_proximity' in layers:
        scored = score_water_proximity(layers['water_proximity'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['water_proximity'])
        print(f"  Water Proximity: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.8, 0))
        weights.append(HABITAT_WEIGHTS['water_proximity'])
        print(f"  Water Proximity: using default (0.8)")

    # Forest Age
    if 'forest_age' in layers:
        scored = score_forest_age(layers['forest_age'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(HABITAT_WEIGHTS['forest_age'])
        print(f"  Forest Age: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.8, 0))
        weights.append(HABITAT_WEIGHTS['forest_age'])
        print(f"  Forest Age: using default (0.8)")

    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    print(f"\nNormalized weights sum: {sum(weights):.4f}")

    # Weighted overlay
    print("\nCalculating weighted overlay...")
    habitat_score = weighted_overlay(scored_layers, weights)

    # Set invalid areas to nodata
    habitat_score = np.where(combined_valid, habitat_score, NODATA)

    return habitat_score, combined_valid


def calculate_access_quality():
    """Calculate access quality score."""
    print("\n" + "-" * 40)
    print("CALCULATING ACCESS QUALITY")
    print("-" * 40)

    # Load access layers
    print("\nLoading layers...")
    layers = {}
    valid_masks = []

    for name in ['road_distance', 'road_type', 'land_ownership']:
        data, mask = load_layer(name, required=False)
        if data is not None:
            layers[name] = data
            valid_masks.append(mask)
            print(f"  ✓ {name}")
        else:
            print(f"  - {name} (will use default)")

    if not valid_masks:
        print("No access layers found. Skipping access calculation.")
        return None, None

    # Combined valid mask
    combined_valid = valid_masks[0].copy()
    for mask in valid_masks[1:]:
        combined_valid &= mask

    # Apply scoring functions
    print("\nScoring layers...")
    scored_layers = []
    weights = []

    # Road Distance
    if 'road_distance' in layers:
        scored = score_road_distance(layers['road_distance'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(ACCESS_WEIGHTS['road_distance'])
        print(f"  Road Distance: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.5, 0))
        weights.append(ACCESS_WEIGHTS['road_distance'])

    # Road Type
    if 'road_type' in layers:
        scored = score_road_type(layers['road_type'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(ACCESS_WEIGHTS['road_type'])
        print(f"  Road Type: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.5, 0))
        weights.append(ACCESS_WEIGHTS['road_type'])

    # Land Ownership
    if 'land_ownership' in layers:
        scored = score_land_ownership(layers['land_ownership'])
        scored_layers.append(np.where(combined_valid, scored, 0))
        weights.append(ACCESS_WEIGHTS['land_ownership'])
        print(f"  Land Ownership: mean score = {np.nanmean(scored[combined_valid]):.3f}")
    else:
        scored_layers.append(np.where(combined_valid, 0.5, 0))
        weights.append(ACCESS_WEIGHTS['land_ownership'])

    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    # Weighted overlay
    print("\nCalculating weighted overlay...")
    access_score = weighted_overlay(scored_layers, weights)

    # Set invalid areas to nodata
    access_score = np.where(combined_valid, access_score, NODATA)

    return access_score, combined_valid


def calculate_suitability():
    """Main suitability calculation workflow."""
    print("=" * 60)
    print("SUITABILITY ANALYSIS")
    print("=" * 60)

    # Get metadata from template
    template_path = PROCESSED_DIR / "elevation_ft.tif"
    if not template_path.exists():
        print("ERROR: Template raster not found. Run preprocessing scripts first.")
        return

    _, meta = read_raster(template_path)

    # Calculate habitat suitability
    habitat_score, habitat_valid = calculate_habitat_suitability()

    # Save habitat suitability
    habitat_path = OUTPUT_DIR / "habitat_suitability.tif"
    write_raster(habitat_path, habitat_score, meta['transform'], TARGET_CRS)
    print(f"\n✓ Habitat suitability saved to: {habitat_path}")

    # Print statistics
    stats = get_raster_stats(habitat_path)
    print(f"\nHabitat Suitability Statistics:")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    print(f"  Mean: {stats['mean']:.3f}")

    # Distribution
    valid_scores = habitat_score[habitat_valid]
    print(f"\nScore Distribution:")
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        pct = np.sum(valid_scores >= threshold) / len(valid_scores) * 100
        print(f"  >= {threshold}: {pct:.1f}%")

    # Calculate access quality
    access_score, access_valid = calculate_access_quality()

    if access_score is not None:
        access_path = OUTPUT_DIR / "access_quality.tif"
        write_raster(access_path, access_score, meta['transform'], TARGET_CRS)
        print(f"\n✓ Access quality saved to: {access_path}")

        stats = get_raster_stats(access_path)
        print(f"\nAccess Quality Statistics:")
        print(f"  Min: {stats['min']:.3f}")
        print(f"  Max: {stats['max']:.3f}")
        print(f"  Mean: {stats['mean']:.3f}")

    # Create combined score (habitat * access) for reference
    if access_score is not None:
        combined_valid = habitat_valid & access_valid
        combined_score = np.where(
            combined_valid,
            habitat_score * access_score,
            NODATA
        )
        combined_path = OUTPUT_DIR / "combined_score.tif"
        write_raster(combined_path, combined_score, meta['transform'], TARGET_CRS)
        print(f"\n✓ Combined score saved to: {combined_path}")

    print("\n" + "=" * 60)
    print("Suitability analysis complete!")
    print("=" * 60)

    # Summary
    print("\nOutput files:")
    for f in OUTPUT_DIR.glob("*.tif"):
        print(f"  - {f.name}")


def main():
    """Entry point."""
    calculate_suitability()


if __name__ == "__main__":
    main()
