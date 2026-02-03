"""
Scoring functions for chanterelle habitat suitability analysis.

Each function converts raw environmental data to a 0-1 suitability score
where 0 = unsuitable and 1 = optimal habitat.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_score(values: np.ndarray, optimal_min: float, optimal_max: float,
                   falloff_low: float, falloff_high: float) -> np.ndarray:
    """
    Apply a Gaussian-like scoring function with optimal range plateau.

    Args:
        values: Input array of values
        optimal_min: Lower bound of optimal range (score = 1.0)
        optimal_max: Upper bound of optimal range (score = 1.0)
        falloff_low: Lower value where score approaches 0
        falloff_high: Upper value where score approaches 0

    Returns:
        Array of scores from 0 to 1
    """
    scores = np.zeros_like(values, dtype=np.float32)

    # Handle nodata
    valid_mask = ~np.isnan(values) & (values != -9999)

    # Optimal range gets score of 1.0
    optimal_mask = valid_mask & (values >= optimal_min) & (values <= optimal_max)
    scores[optimal_mask] = 1.0

    # Below optimal - gaussian falloff
    below_mask = valid_mask & (values < optimal_min)
    if np.any(below_mask):
        sigma = (optimal_min - falloff_low) / 3  # 3 sigma covers most of range
        if sigma > 0:
            scores[below_mask] = np.exp(-0.5 * ((values[below_mask] - optimal_min) / sigma) ** 2)

    # Above optimal - gaussian falloff
    above_mask = valid_mask & (values > optimal_max)
    if np.any(above_mask):
        sigma = (falloff_high - optimal_max) / 3
        if sigma > 0:
            scores[above_mask] = np.exp(-0.5 * ((values[above_mask] - optimal_max) / sigma) ** 2)

    return np.clip(scores, 0, 1)


def score_elevation(elevation_ft: np.ndarray) -> np.ndarray:
    """
    Score elevation for chanterelle habitat.

    Optimal: 2,000-3,000 ft
    Suitable: 1,500-4,000 ft
    Drops off below 1,000 ft and above 4,500 ft
    """
    return gaussian_score(
        elevation_ft,
        optimal_min=2000,
        optimal_max=3000,
        falloff_low=500,
        falloff_high=5000
    )


def score_canopy_cover(canopy_pct: np.ndarray) -> np.ndarray:
    """
    Score tree canopy cover percentage.

    Optimal: 50-70% (partial shade, not dense)
    Drops off below 30% and above 85%
    """
    return gaussian_score(
        canopy_pct,
        optimal_min=50,
        optimal_max=70,
        falloff_low=20,
        falloff_high=95
    )


def score_soil_ph(ph: np.ndarray) -> np.ndarray:
    """
    Score soil pH for chanterelle habitat.

    Optimal: 4.0-5.5 (acidic)
    Tapers to 0 outside 3.5-6.5
    """
    return gaussian_score(
        ph,
        optimal_min=4.0,
        optimal_max=5.5,
        falloff_low=3.0,
        falloff_high=7.0
    )


def score_precipitation(precip_inches: np.ndarray) -> np.ndarray:
    """
    Score annual precipitation.

    Higher is generally better up to ~80 inches/year.
    Linear scaling with plateau at high end.
    """
    scores = np.zeros_like(precip_inches, dtype=np.float32)
    valid_mask = ~np.isnan(precip_inches) & (precip_inches >= 0)

    # Minimum threshold - too dry below 30"
    min_precip = 30
    optimal_precip = 80

    # Below minimum
    below_min = valid_mask & (precip_inches < min_precip)
    scores[below_min] = precip_inches[below_min] / min_precip * 0.3

    # Linear scaling from minimum to optimal
    scaling_mask = valid_mask & (precip_inches >= min_precip) & (precip_inches <= optimal_precip)
    scores[scaling_mask] = 0.3 + 0.7 * (precip_inches[scaling_mask] - min_precip) / (optimal_precip - min_precip)

    # Above optimal - plateau at 1.0
    above_optimal = valid_mask & (precip_inches > optimal_precip)
    scores[above_optimal] = 1.0

    return np.clip(scores, 0, 1)


def score_slope(slope_degrees: np.ndarray) -> np.ndarray:
    """
    Score slope for accessibility and habitat.

    Flat to moderate slopes are fine.
    Penalty for steep slopes (>30 degrees) - difficult terrain and erosion.
    """
    scores = np.ones_like(slope_degrees, dtype=np.float32)
    valid_mask = ~np.isnan(slope_degrees) & (slope_degrees >= 0)

    # Mild slopes (0-15) = 1.0
    # Moderate slopes (15-30) = slight reduction
    # Steep slopes (>30) = significant penalty

    moderate = valid_mask & (slope_degrees > 15) & (slope_degrees <= 30)
    scores[moderate] = 1.0 - 0.2 * (slope_degrees[moderate] - 15) / 15

    steep = valid_mask & (slope_degrees > 30)
    scores[steep] = 0.8 - 0.6 * np.minimum((slope_degrees[steep] - 30) / 30, 1.0)

    return np.clip(scores, 0.2, 1)


def score_aspect(aspect_degrees: np.ndarray) -> np.ndarray:
    """
    Score slope aspect (direction facing).

    North-facing slopes (315-360, 0-45) are optimal - cooler, moister.
    South-facing slopes are less suitable.
    """
    scores = np.zeros_like(aspect_degrees, dtype=np.float32)
    valid_mask = ~np.isnan(aspect_degrees) & (aspect_degrees >= 0)

    # Convert aspect to "northness" (-1 to 1, where 1 = north)
    # North = 0/360, South = 180
    aspect_rad = np.radians(aspect_degrees)
    northness = np.cos(aspect_rad)  # 1 at north, -1 at south

    # Scale to 0-1 where north = 1, south = 0.5
    scores[valid_mask] = 0.5 + 0.5 * northness[valid_mask]

    # Flat areas (aspect = -1 or undefined) get neutral score
    flat_mask = aspect_degrees < 0
    scores[flat_mask] = 0.75

    return np.clip(scores, 0.5, 1)


def score_water_proximity(distance_m: np.ndarray) -> np.ndarray:
    """
    Score proximity to water features (streams, rivers).

    Bonus within 500m of streams (moisture availability).
    Neutral beyond 500m.
    Slight penalty if too close (<50m, might be too wet).
    """
    scores = np.ones_like(distance_m, dtype=np.float32) * 0.7
    valid_mask = ~np.isnan(distance_m) & (distance_m >= 0)

    # Very close to water - slightly too wet
    very_close = valid_mask & (distance_m < 50)
    scores[very_close] = 0.8

    # Optimal range - 50-500m
    optimal = valid_mask & (distance_m >= 50) & (distance_m <= 500)
    scores[optimal] = 1.0

    # Farther away - gradual decrease
    far = valid_mask & (distance_m > 500) & (distance_m <= 2000)
    scores[far] = 1.0 - 0.3 * (distance_m[far] - 500) / 1500

    # Very far - neutral
    very_far = valid_mask & (distance_m > 2000)
    scores[very_far] = 0.7

    return np.clip(scores, 0.7, 1)


def score_forest_age(age_years: np.ndarray) -> np.ndarray:
    """
    Score forest stand age.

    Second-growth (40-80 years) is optimal for chanterelles.
    Old growth (>150 years) is good but slightly less productive.
    Young forests (<20 years) are not suitable.
    """
    scores = np.zeros_like(age_years, dtype=np.float32)
    valid_mask = ~np.isnan(age_years) & (age_years >= 0)

    # Young forest - not suitable
    young = valid_mask & (age_years < 20)
    scores[young] = 0.1

    # Maturing forest - improving
    maturing = valid_mask & (age_years >= 20) & (age_years < 40)
    scores[maturing] = 0.3 + 0.5 * (age_years[maturing] - 20) / 20

    # Second growth - optimal
    second_growth = valid_mask & (age_years >= 40) & (age_years <= 80)
    scores[second_growth] = 1.0

    # Mature forest - still good
    mature = valid_mask & (age_years > 80) & (age_years <= 150)
    scores[mature] = 1.0 - 0.2 * (age_years[mature] - 80) / 70

    # Old growth - good but less productive
    old_growth = valid_mask & (age_years > 150)
    scores[old_growth] = 0.7

    return np.clip(scores, 0.1, 1)


def score_forest_type(forest_code: np.ndarray, target_codes: set) -> np.ndarray:
    """
    Score forest type based on mycorrhizal associations.

    Args:
        forest_code: Array of LANDFIRE EVT codes
        target_codes: Set of codes for Douglas-fir, Western Hemlock, Spruce

    Returns:
        Suitability scores (1.0 for target species, 0.3 for other conifers, 0 for non-forest)
    """
    scores = np.zeros_like(forest_code, dtype=np.float32)

    # Target mycorrhizal species get 1.0
    for code in target_codes:
        scores[forest_code == code] = 1.0

    return scores


# LANDFIRE EVT codes for Pacific Northwest forests
# These are approximate - actual codes should be verified from LANDFIRE documentation
LANDFIRE_DOUGLAS_FIR_CODES = {
    7051, 7052, 7053,  # Douglas-fir types
    7111, 7112, 7113,  # Douglas-fir/Western Hemlock
}

LANDFIRE_HEMLOCK_CODES = {
    7061, 7062, 7063,  # Western Hemlock types
    7071, 7072, 7073,  # Mountain Hemlock types
}

LANDFIRE_SPRUCE_CODES = {
    7081, 7082, 7083,  # Sitka Spruce types
    7091, 7092, 7093,  # Engelmann Spruce types
}

LANDFIRE_OTHER_CONIFER_CODES = {
    7021, 7022, 7023,  # Western Red Cedar
    7031, 7032, 7033,  # Pacific Silver Fir
    7041, 7042, 7043,  # Grand Fir
}

# All target forest types for chanterelles
CHANTERELLE_FOREST_CODES = (
    LANDFIRE_DOUGLAS_FIR_CODES |
    LANDFIRE_HEMLOCK_CODES |
    LANDFIRE_SPRUCE_CODES
)


def score_road_distance(distance_m: np.ndarray) -> np.ndarray:
    """
    Score proximity to roads for accessibility.

    Closer to roads = better access.
    Inverse distance decay.
    """
    scores = np.zeros_like(distance_m, dtype=np.float32)
    valid_mask = ~np.isnan(distance_m) & (distance_m >= 0)

    # Within 100m - excellent access
    close = valid_mask & (distance_m <= 100)
    scores[close] = 1.0

    # 100m - 1km - good access with decay
    medium = valid_mask & (distance_m > 100) & (distance_m <= 1000)
    scores[medium] = 1.0 - 0.5 * (distance_m[medium] - 100) / 900

    # 1km - 5km - moderate access
    far = valid_mask & (distance_m > 1000) & (distance_m <= 5000)
    scores[far] = 0.5 - 0.4 * (distance_m[far] - 1000) / 4000

    # Beyond 5km - poor access
    very_far = valid_mask & (distance_m > 5000)
    scores[very_far] = 0.1 * np.exp(-(distance_m[very_far] - 5000) / 5000)

    return np.clip(scores, 0, 1)


def score_road_type(road_type: np.ndarray) -> np.ndarray:
    """
    Score road type/quality.

    1 = Paved road (score 1.0)
    2 = Unpaved/gravel (score 0.7)
    3 = Trail/track (score 0.3)
    0 = Unknown (score 0.5)
    """
    scores = np.ones_like(road_type, dtype=np.float32) * 0.5

    scores[road_type == 1] = 1.0  # Paved
    scores[road_type == 2] = 0.7  # Unpaved
    scores[road_type == 3] = 0.3  # Trail

    return scores


def score_land_ownership(ownership_code: np.ndarray) -> np.ndarray:
    """
    Score land ownership for legal foraging access.

    1 = National Forest (legal foraging, score 1.0)
    2 = State land (usually allowed, score 0.8)
    3 = Private (not accessible, score 0.2)
    0 = Unknown (score 0.5)
    """
    scores = np.ones_like(ownership_code, dtype=np.float32) * 0.5

    scores[ownership_code == 1] = 1.0  # National Forest
    scores[ownership_code == 2] = 0.8  # State land
    scores[ownership_code == 3] = 0.2  # Private

    return scores


def weighted_overlay(layers: list, weights: list) -> np.ndarray:
    """
    Perform weighted overlay of multiple scored layers.

    Args:
        layers: List of numpy arrays (all same shape, 0-1 scaled)
        weights: List of weights (should sum to 1.0)

    Returns:
        Composite suitability score array (0-1)
    """
    if len(layers) != len(weights):
        raise ValueError("Number of layers must match number of weights")

    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError(f"Weights should sum to 1.0, got {sum(weights)}")

    # Stack layers and apply weights
    result = np.zeros_like(layers[0], dtype=np.float32)

    for layer, weight in zip(layers, weights):
        result += layer * weight

    return np.clip(result, 0, 1)
