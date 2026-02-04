"""
Script 08: Export rasters for Mapbox visualization.

This script:
1. Converts suitability rasters to Cloud-Optimized GeoTIFF (COG)
2. Optionally creates MBTiles for Mapbox upload
3. Generates Mapbox style configuration
4. Creates preview images

Output formats:
- Cloud-Optimized GeoTIFF (COG) - for Mapbox raster tiles
- MBTiles - alternative format for Mapbox upload
- PNG preview - for quick visual check
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.raster_utils import (
    read_raster,
    write_raster,
    create_cog,
    get_raster_stats,
    NODATA,
)

# Directories
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MAPBOX_DIR = PROJECT_ROOT / "mapbox"
MAPBOX_DIR.mkdir(parents=True, exist_ok=True)

# Color ramp for chanterelle heat map (golden/orange theme)
# Values are [score_threshold, R, G, B, A]
HABITAT_COLOR_RAMP = [
    [0.0, 128, 0, 128, 0],      # Transparent purple (unsuitable)
    [0.2, 75, 0, 130, 180],     # Dark purple
    [0.4, 138, 43, 226, 200],   # Blue-violet
    [0.5, 255, 140, 0, 220],    # Dark orange
    [0.6, 255, 165, 0, 230],    # Orange
    [0.7, 255, 200, 0, 240],    # Gold-orange
    [0.8, 255, 215, 0, 250],    # Gold
    [0.9, 255, 235, 100, 255],  # Bright gold
    [1.0, 255, 255, 150, 255],  # Pale gold (optimal)
]

ACCESS_COLOR_RAMP = [
    [0.0, 0, 0, 0, 0],          # Transparent (no access)
    [0.2, 50, 50, 50, 150],     # Dark gray
    [0.4, 100, 100, 100, 180],  # Gray
    [0.6, 150, 200, 150, 200],  # Light green
    [0.8, 100, 220, 100, 220],  # Green
    [1.0, 50, 255, 50, 255],    # Bright green (excellent access)
]


def convert_to_cog(input_path: Path, output_path: Path):
    """
    Convert a raster to Cloud-Optimized GeoTIFF (8-bit for Mapbox).

    Mapbox requires 8-bit TIFFs. This converts 0-1 float scores to 0-255 uint8.

    Args:
        input_path: Input GeoTIFF path (float32, 0-1 range)
        output_path: Output COG path (uint8, 0-255 range)
    """
    print(f"Converting to COG: {input_path.name}...")

    with rasterio.open(input_path) as src:
        data = src.read(1)
        nodata = src.nodata

        # Create valid mask
        if nodata is not None:
            valid_mask = ~np.isnan(data) & (data != nodata)
        else:
            valid_mask = ~np.isnan(data)

        # Convert float (0-1) to uint8 (0-255)
        # Use 1-255 for valid data, 0 for nodata
        data_uint8 = np.zeros(data.shape, dtype=np.uint8)
        data_uint8[valid_mask] = np.clip(data[valid_mask] * 254 + 1, 1, 255).astype(np.uint8)
        # 0 = nodata, 1 = score 0.0, 255 = score 1.0

        # Update profile for 8-bit COG
        profile = src.profile.copy()
        profile.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'compress': 'deflate',
            'predictor': 2,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
        })

        # Write COG with overviews
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data_uint8, 1)

            # Build overviews for faster rendering at different zoom levels
            overview_levels = [2, 4, 8, 16, 32]
            dst.build_overviews(overview_levels, Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    # Report size reduction
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Created: {output_path}")
    print(f"  Size: {input_size:.1f} MB -> {output_size:.1f} MB (8-bit)")


def normalize_for_visualization(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Normalize data to 0-255 range for visualization.

    Assumes input is 0-1 suitability score.
    """
    # Clip to 0-1 range
    normalized = np.clip(data, 0, 1)

    # Scale to 0-255
    scaled = (normalized * 255).astype(np.uint8)

    # Set nodata areas to 0
    scaled[~valid_mask] = 0

    return scaled


def create_preview_png(input_path: Path, output_path: Path, color_ramp: list):
    """
    Create a preview PNG image with color ramp applied.

    Args:
        input_path: Input suitability raster
        output_path: Output PNG path
        color_ramp: List of [threshold, R, G, B, A] values
    """
    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available for PNG preview")
        return

    print(f"Creating preview: {output_path.name}...")

    data, meta = read_raster(input_path)
    nodata = meta.get('nodata', NODATA)

    # Create valid mask
    valid_mask = ~np.isnan(data) & (data != nodata)

    # Create RGBA image
    height, width = data.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Apply color ramp
    for i in range(len(color_ramp) - 1):
        low_thresh, low_r, low_g, low_b, low_a = color_ramp[i]
        high_thresh, high_r, high_g, high_b, high_a = color_ramp[i + 1]

        # Find pixels in this range
        mask = valid_mask & (data >= low_thresh) & (data < high_thresh)

        if np.any(mask):
            # Interpolate color within range
            t = (data[mask] - low_thresh) / (high_thresh - low_thresh + 1e-10)

            rgba[mask, 0] = (low_r + t * (high_r - low_r)).astype(np.uint8)
            rgba[mask, 1] = (low_g + t * (high_g - low_g)).astype(np.uint8)
            rgba[mask, 2] = (low_b + t * (high_b - low_b)).astype(np.uint8)
            rgba[mask, 3] = (low_a + t * (high_a - low_a)).astype(np.uint8)

    # Handle top of range
    top_thresh, top_r, top_g, top_b, top_a = color_ramp[-1]
    top_mask = valid_mask & (data >= top_thresh)
    rgba[top_mask] = [top_r, top_g, top_b, top_a]

    # Create and save image
    img = Image.fromarray(rgba, 'RGBA')

    # Resize for preview (max 2000px)
    max_size = 2000
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    img.save(output_path)
    print(f"  Created: {output_path}")


def generate_mapbox_style():
    """
    Generate Mapbox style JSON configuration.

    This creates a dark-themed map style with the heat map layers.
    """
    style = {
        "version": 8,
        "name": "Chanterelle Habitat Map",
        "metadata": {
            "description": "Washington Cascadia chanterelle habitat suitability map"
        },
        "sources": {
            "habitat": {
                "type": "raster",
                "tiles": [
                    "YOUR_MAPBOX_TILESET_URL/{z}/{x}/{y}.png"
                ],
                "tileSize": 256,
                "attribution": "Chanterelle Finder Project"
            },
            "access": {
                "type": "raster",
                "tiles": [
                    "YOUR_MAPBOX_ACCESS_TILESET_URL/{z}/{x}/{y}.png"
                ],
                "tileSize": 256
            }
        },
        "sprite": "mapbox://sprites/mapbox/dark-v11",
        "glyphs": "mapbox://fonts/mapbox/{fontstack}/{range}.pbf",
        "layers": [
            {
                "id": "background",
                "type": "background",
                "paint": {
                    "background-color": "#1a1a2e"
                }
            },
            {
                "id": "habitat-suitability",
                "type": "raster",
                "source": "habitat",
                "paint": {
                    "raster-opacity": 0.8,
                    "raster-resampling": "linear"
                },
                "layout": {
                    "visibility": "visible"
                }
            },
            {
                "id": "access-quality",
                "type": "raster",
                "source": "access",
                "paint": {
                    "raster-opacity": 0.6,
                    "raster-resampling": "linear"
                },
                "layout": {
                    "visibility": "none"
                }
            }
        ]
    }

    return style


def generate_legend_config():
    """Generate legend configuration for the map."""
    legend = {
        "habitat": {
            "title": "Chanterelle Habitat Suitability",
            "type": "gradient",
            "colors": [
                {"value": 0.0, "color": "#800080", "label": "Unsuitable"},
                {"value": 0.3, "color": "#8a2be2", "label": "Low"},
                {"value": 0.5, "color": "#ff8c00", "label": "Moderate"},
                {"value": 0.7, "color": "#ffc800", "label": "Good"},
                {"value": 0.9, "color": "#ffeb64", "label": "Excellent"},
            ]
        },
        "access": {
            "title": "Access Quality",
            "type": "gradient",
            "colors": [
                {"value": 0.0, "color": "#323232", "label": "No Access"},
                {"value": 0.5, "color": "#96c896", "label": "Moderate"},
                {"value": 1.0, "color": "#32ff32", "label": "Easy Access"},
            ]
        }
    }

    return legend


def export_for_mapbox():
    """Main export workflow."""
    print("=" * 60)
    print("EXPORTING FOR MAPBOX")
    print("=" * 60)

    # Check for input files
    habitat_path = OUTPUT_DIR / "habitat_suitability.tif"
    access_path = OUTPUT_DIR / "access_quality.tif"

    if not habitat_path.exists():
        print("ERROR: Habitat suitability raster not found.")
        print("Run 07_calculate_suitability.py first.")
        return

    # Export COGs
    print("\n[1/4] Creating Cloud-Optimized GeoTIFFs...")

    habitat_cog = MAPBOX_DIR / "habitat_suitability_cog.tif"
    convert_to_cog(habitat_path, habitat_cog)

    if access_path.exists():
        access_cog = MAPBOX_DIR / "access_quality_cog.tif"
        convert_to_cog(access_path, access_cog)

    # Create preview images
    print("\n[2/4] Creating preview images...")

    habitat_preview = MAPBOX_DIR / "habitat_preview.png"
    create_preview_png(habitat_path, habitat_preview, HABITAT_COLOR_RAMP)

    if access_path.exists():
        access_preview = MAPBOX_DIR / "access_preview.png"
        create_preview_png(access_path, access_preview, ACCESS_COLOR_RAMP)

    # Generate Mapbox style
    print("\n[3/4] Generating Mapbox style configuration...")

    style = generate_mapbox_style()
    style_path = MAPBOX_DIR / "style.json"
    with open(style_path, 'w') as f:
        json.dump(style, f, indent=2)
    print(f"  Created: {style_path}")

    # Generate legend config
    legend = generate_legend_config()
    legend_path = MAPBOX_DIR / "legend.json"
    with open(legend_path, 'w') as f:
        json.dump(legend, f, indent=2)
    print(f"  Created: {legend_path}")

    # Print upload instructions
    print("\n[4/4] Mapbox Upload Instructions")
    print("-" * 40)
    print("""
To upload to Mapbox:

1. Install Mapbox CLI:
   npm install -g @mapbox/mapbox-cli
   # or
   pip install mapboxcli

2. Authenticate:
   mapbox config set access_token YOUR_ACCESS_TOKEN

3. Upload tilesets:
   mapbox upload username.chanterelle-habitat mapbox/habitat_suitability_cog.tif
   mapbox upload username.chanterelle-access mapbox/access_quality_cog.tif

4. Update style.json with your tileset URLs

5. Create a Mapbox style using the style.json configuration

Alternative: Use Mapbox Studio
- Go to https://studio.mapbox.com/
- Create new tileset by uploading COG files
- Create new style using Dark template
- Add raster layers for habitat and access

Note: COGs are 8-bit (Mapbox requirement)
- Value 0 = nodata (transparent)
- Value 1-255 = suitability score (1=0%, 255=100%)
- To convert back: score = (pixel_value - 1) / 254

COG files ready for upload:
""")

    for f in MAPBOX_DIR.glob("*_cog.tif"):
        stats = get_raster_stats(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


def main():
    """Entry point."""
    export_for_mapbox()


if __name__ == "__main__":
    main()
