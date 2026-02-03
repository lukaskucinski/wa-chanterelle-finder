"""
Raster processing utilities for chanterelle habitat analysis.

Provides functions for:
- Reading and writing rasters
- Reprojection and resampling
- Clipping to study area
- Distance calculations
- DEM derivatives (slope, aspect)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from shapely.geometry import box


# Project constants
TARGET_CRS = CRS.from_epsg(32610)  # UTM 10N
TARGET_RESOLUTION = 30  # meters
NODATA = -9999


def get_study_area_bounds() -> dict:
    """
    Return the bounding box for the Cascade Range study area.

    Approximate bounds covering the Cascade Range in Washington State.
    """
    return {
        'west': -122.5,
        'east': -120.5,
        'south': 45.5,
        'north': 49.0,
    }


def create_study_area_polygon(bounds: dict = None) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with the study area polygon."""
    if bounds is None:
        bounds = get_study_area_bounds()

    geometry = box(bounds['west'], bounds['south'], bounds['east'], bounds['north'])
    gdf = gpd.GeoDataFrame({'geometry': [geometry]}, crs='EPSG:4326')
    return gdf


def read_raster(filepath: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Read a raster file and return data with metadata.

    Args:
        filepath: Path to raster file

    Returns:
        Tuple of (data array, metadata dict including transform, crs, etc.)
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)
        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'nodata': src.nodata,
            'dtype': src.dtypes[0],
            'bounds': src.bounds,
        }
    return data, meta


def write_raster(
    filepath: Union[str, Path],
    data: np.ndarray,
    transform,
    crs,
    nodata: float = NODATA,
    dtype: str = 'float32'
) -> None:
    """
    Write a numpy array to a raster file.

    Args:
        filepath: Output path
        data: 2D numpy array
        transform: Affine transform
        crs: Coordinate reference system
        nodata: NoData value
        dtype: Data type string
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'width': data.shape[1],
        'height': data.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'nodata': nodata,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }

    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)


def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    dst_crs: CRS = TARGET_CRS,
    dst_resolution: float = TARGET_RESOLUTION,
    resampling: Resampling = Resampling.bilinear
) -> None:
    """
    Reproject a raster to the target CRS and resolution.

    Args:
        src_path: Input raster path
        dst_path: Output raster path
        dst_crs: Target CRS
        dst_resolution: Target resolution in CRS units
        resampling: Resampling method
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=dst_resolution
        )

        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw',
        })

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling
                )


def clip_raster_to_bounds(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    bounds: dict = None,
    buffer_m: float = 5000
) -> None:
    """
    Clip a raster to the study area bounds.

    Args:
        src_path: Input raster path
        dst_path: Output raster path
        bounds: Dict with west, east, south, north keys (WGS84)
        buffer_m: Buffer distance in meters
    """
    if bounds is None:
        bounds = get_study_area_bounds()

    # Create geometry with buffer
    study_area = create_study_area_polygon(bounds)
    study_area_utm = study_area.to_crs(TARGET_CRS)
    study_area_buffered = study_area_utm.buffer(buffer_m)

    with rasterio.open(src_path) as src:
        # Reproject clip geometry to source CRS if needed
        clip_geom = study_area_buffered.to_crs(src.crs)

        out_image, out_transform = mask(
            src,
            clip_geom.geometry,
            crop=True,
            nodata=src.nodata or NODATA
        )

        profile = src.profile.copy()
        profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform,
            'compress': 'lzw',
        })

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(out_image)


def align_raster_to_template(
    src_path: Union[str, Path],
    template_path: Union[str, Path],
    dst_path: Union[str, Path],
    resampling: Resampling = Resampling.bilinear
) -> None:
    """
    Align a raster to match a template raster's grid.

    Args:
        src_path: Input raster to align
        template_path: Template raster with target grid
        dst_path: Output aligned raster
        resampling: Resampling method
    """
    with rasterio.open(template_path) as template:
        template_transform = template.transform
        template_crs = template.crs
        template_width = template.width
        template_height = template.height

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({
            'crs': template_crs,
            'transform': template_transform,
            'width': template_width,
            'height': template_height,
            'compress': 'lzw',
        })

        dst_array = np.empty((template_height, template_width), dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_transform,
            dst_crs=template_crs,
            resampling=resampling
        )

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(dst_array, 1)


def calculate_slope(dem: np.ndarray, resolution: float = TARGET_RESOLUTION) -> np.ndarray:
    """
    Calculate slope in degrees from a DEM.

    Args:
        dem: 2D numpy array of elevations
        resolution: Cell size in map units

    Returns:
        Slope in degrees
    """
    # Handle nodata
    dem_masked = np.ma.masked_equal(dem, NODATA)
    dem_masked = np.ma.masked_invalid(dem_masked)

    # Calculate gradients
    dy, dx = np.gradient(dem_masked, resolution)

    # Calculate slope
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Restore nodata
    slope_deg = np.where(dem_masked.mask, NODATA, slope_deg)

    return slope_deg.astype(np.float32)


def calculate_aspect(dem: np.ndarray, resolution: float = TARGET_RESOLUTION) -> np.ndarray:
    """
    Calculate aspect in degrees from a DEM.

    Returns degrees clockwise from north (0-360).
    Flat areas return -1.

    Args:
        dem: 2D numpy array of elevations
        resolution: Cell size in map units

    Returns:
        Aspect in degrees (0-360, -1 for flat)
    """
    # Handle nodata
    dem_masked = np.ma.masked_equal(dem, NODATA)
    dem_masked = np.ma.masked_invalid(dem_masked)

    # Calculate gradients
    dy, dx = np.gradient(dem_masked, resolution)

    # Calculate aspect
    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad)

    # Convert to 0-360 range (clockwise from north)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

    # Mark flat areas
    slope_mag = np.sqrt(dx**2 + dy**2)
    flat_mask = slope_mag < 0.01
    aspect_deg = np.where(flat_mask, -1, aspect_deg)

    # Restore nodata
    aspect_deg = np.where(dem_masked.mask, NODATA, aspect_deg)

    return aspect_deg.astype(np.float32)


def rasterize_vector(
    gdf: gpd.GeoDataFrame,
    template_path: Union[str, Path],
    attribute: str = None,
    fill_value: float = 0,
    dtype: str = 'float32'
) -> np.ndarray:
    """
    Rasterize a vector dataset to match a template raster.

    Args:
        gdf: GeoDataFrame to rasterize
        template_path: Template raster for extent and resolution
        attribute: Attribute column to burn into raster (None for binary)
        fill_value: Background fill value
        dtype: Output data type

    Returns:
        Rasterized numpy array
    """
    with rasterio.open(template_path) as template:
        out_shape = (template.height, template.width)
        transform = template.transform
        crs = template.crs

    # Reproject vector if needed
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Prepare shapes
    if attribute:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
    else:
        shapes = ((geom, 1) for geom in gdf.geometry)

    rasterized = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill_value,
        dtype=dtype
    )

    return rasterized


def calculate_distance_raster(
    binary_raster: np.ndarray,
    resolution: float = TARGET_RESOLUTION
) -> np.ndarray:
    """
    Calculate Euclidean distance from features in a binary raster.

    Args:
        binary_raster: Array where features = 1, background = 0
        resolution: Cell size in map units

    Returns:
        Distance raster in map units
    """
    # Invert for distance_transform (0 = feature, 1 = background)
    inverted = (binary_raster == 0).astype(np.float32)

    # Calculate distance
    distance = distance_transform_edt(inverted) * resolution

    return distance.astype(np.float32)


def meters_to_feet(meters: np.ndarray) -> np.ndarray:
    """Convert meters to feet."""
    return meters * 3.28084


def feet_to_meters(feet: np.ndarray) -> np.ndarray:
    """Convert feet to meters."""
    return feet / 3.28084


def create_cog(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    overview_levels: list = None
) -> None:
    """
    Convert a GeoTIFF to Cloud-Optimized GeoTIFF (COG).

    Args:
        src_path: Input GeoTIFF path
        dst_path: Output COG path
        overview_levels: Overview levels (e.g., [2, 4, 8, 16])
    """
    if overview_levels is None:
        overview_levels = [2, 4, 8, 16, 32]

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        data = src.read()

        profile.update({
            'driver': 'GTiff',
            'compress': 'deflate',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'interleave': 'pixel',
        })

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data)

            # Build overviews
            dst.build_overviews(overview_levels, Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')


def get_raster_stats(filepath: Union[str, Path]) -> dict:
    """
    Get basic statistics for a raster file.

    Args:
        filepath: Path to raster file

    Returns:
        Dict with min, max, mean, std, nodata_count
    """
    with rasterio.open(filepath) as src:
        data = src.read(1)
        nodata = src.nodata

    valid_mask = ~np.isnan(data)
    if nodata is not None:
        valid_mask &= (data != nodata)

    valid_data = data[valid_mask]

    return {
        'min': float(np.min(valid_data)) if len(valid_data) > 0 else None,
        'max': float(np.max(valid_data)) if len(valid_data) > 0 else None,
        'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else None,
        'std': float(np.std(valid_data)) if len(valid_data) > 0 else None,
        'nodata_count': int(np.sum(~valid_mask)),
        'valid_count': int(np.sum(valid_mask)),
    }
