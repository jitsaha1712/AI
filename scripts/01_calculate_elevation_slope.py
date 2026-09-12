import glob
import os
import zipfile
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import pyogrio
from rasterstats import zonal_stats

# Define paths relative to project root
DEM_DIR = os.path.join("Data", "dem")
OSM_DIR = os.path.join("Data", "osm")
PROCESSED_DIR = os.path.join("Data", "processed")

# Ensure 'processed' directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Output paths
MERGED_DEM = os.path.join(PROCESSED_DIR, "dem_merged_wgs84.tif")
UTM_DEM = os.path.join(PROCESSED_DIR, "elevation_utm.tif")
UTM_SLOPE = os.path.join(PROCESSED_DIR, "slope_utm.tif")
OUTPUT_ROADS = os.path.join(PROCESSED_DIR, "roads_with_elevation_slope.gpkg")

# Target CRS for Northeast India (UTM Zone 45N)
TARGET_CRS = "EPSG:32645"

print("=" * 50)
print("STARTING STEP 3: ELEVATION & SLOPE EXTRACTION")
print("=" * 50)

# --- 1. Mosaic DEM Tiles ---
print("\n[1/4] Merging raw DEM tiles...")
dem_files = (
    glob.glob(os.path.join(DEM_DIR, "*.tif")) + 
    glob.glob(os.path.join(DEM_DIR, "*.dem")) +
    glob.glob(os.path.join(DEM_DIR, "*.img"))
)

if not dem_files:
    raise FileNotFoundError(f"No DEM raster files found in '{DEM_DIR}'!")

print(f"    Found {len(dem_files)} DEM tiles.")
src_files_to_mosaic = [rasterio.open(f) for f in dem_files]

mosaic, out_trans = merge(src_files_to_mosaic)
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "crs": src_files_to_mosaic[0].crs
})

with rasterio.open(MERGED_DEM, "w", **out_meta) as dest:
    dest.write(mosaic)

for src in src_files_to_mosaic:
    src.close()
print(f"    Saved merged DEM to: {MERGED_DEM}")


# --- 2. Reproject DEM to UTM (Meters) ---
print("\n[2/4] Reprojecting merged DEM to UTM Zone 45N (EPSG:32645)...")
with rasterio.open(MERGED_DEM) as src:
    transform, width, height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': TARGET_CRS,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(UTM_DEM, "w", **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear
        )
print(f"    Saved metric DEM to: {UTM_DEM}")


# --- 3. Compute Slope Raster ---
print("\n[3/4] Calculating terrain slope in degrees...")
with rasterio.open(UTM_DEM) as src:
    elevation = src.read(1).astype(np.float32)
    
    # Handle nodata values if present
    nodata = src.nodata
    if nodata is not None:
        elevation[elevation == nodata] = np.nan

    dx = abs(src.transform[0])
    dy = abs(src.transform[4])
    
    # Gradients along X and Y
    py, px = np.gradient(elevation, dy, dx)
    slope_deg = np.arctan(np.sqrt(px**2 + py**2)) * (180.0 / np.pi)
    
    # Fill NaN values back to original nodata or 0 for writing
    if nodata is not None:
        slope_deg = np.nan_to_num(slope_deg, nan=nodata)

    kwargs = src.meta.copy()
    kwargs.update(dtype=rasterio.float32)
    with rasterio.open(UTM_SLOPE, "w", **kwargs) as dst:
        dst.write(slope_deg.astype(rasterio.float32), 1)
print(f"    Saved slope raster to: {UTM_SLOPE}")


# --- 4. Zonal Statistics onto Road Vector ---
print("\n[4/4] Extracting elevation & slope onto OSM roads...")

osm_files = (
    glob.glob(os.path.join(OSM_DIR, "*.gpkg")) + 
    glob.glob(os.path.join(OSM_DIR, "*.zip"))
)

if not osm_files:
    raise FileNotFoundError(f"No OSM file found in '{OSM_DIR}'!")

target_osm = osm_files[0]

# Auto-extract zip file to prevent Windows path issues with pyogrio
if target_osm.endswith(".zip"):
    print(f"    Extracting zipped GeoPackage: {target_osm}...")
    with zipfile.ZipFile(target_osm, 'r') as zip_ref:
        zip_ref.extractall(OSM_DIR)
    
    # Find extracted .gpkg files
    extracted_gpkgs = glob.glob(os.path.join(OSM_DIR, "*.gpkg"))
    if not extracted_gpkgs:
        raise FileNotFoundError(f"No .gpkg file found after unzipping {target_osm}")
    target_osm = extracted_gpkgs[0]

print(f"    Reading road network from: {target_osm}")

# Inspect layers available in the GeoPackage dynamically
layers_info = pyogrio.list_layers(target_osm)
available_layers = [l[0] for l in layers_info]
print(f"    Available layers: {available_layers}")

# Match layer name for roads (handles Geofabrik or generic OSM layers)
target_layer = None
for layer_name in available_layers:
    if "road" in layer_name.lower() or layer_name.lower() == "lines":
        target_layer = layer_name
        break

if target_layer:
    print(f"    Loading layer: '{target_layer}'")
    roads = gpd.read_file(target_osm, layer=target_layer)
else:
    print(f"    No explicit 'roads' or 'lines' layer found. Reading default layer: '{available_layers[0]}'")
    roads = gpd.read_file(target_osm, layer=available_layers[0])

# Convert roads to UTM Zone 45N
print("    Reprojecting roads to UTM Zone 45N...")
roads_utm = roads.to_crs(TARGET_CRS)

# Run zonal statistics
print("    Computing elevation stats per road segment...")
elev_stats = zonal_stats(roads_utm, UTM_DEM, stats=["mean", "max"])

print("    Computing slope stats per road segment...")
slope_stats = zonal_stats(roads_utm, UTM_SLOPE, stats=["mean", "max"])

# Assign new columns
roads["mean_elevation"] = [s["mean"] for s in elev_stats]
roads["max_elevation"] = [s["max"] for s in elev_stats]
roads["mean_slope"] = [s["mean"] for s in slope_stats]
roads["max_slope"] = [s["max"] for s in slope_stats]

# Save output
print(f"    Saving enriched road layer to: {OUTPUT_ROADS}")
roads.to_file(OUTPUT_ROADS, driver="GPKG")

print("\n" + "=" * 50)
print(f"STEP 3 COMPLETED SUCCESSFULLY!")
print(f"Processed roads file created: {OUTPUT_ROADS}")
print("=" * 50)