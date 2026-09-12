import geopandas as gpd
import rasterio
from shapely.geometry import Point

print("Loading road network...")
roads_path = "Data/processed/assam_roads_elevation_slope.gpkg"
roads_df = gpd.read_file(roads_path)

# Ensure matching Coordinate Reference System (CRS) - EPSG:4326 is standard for GEE exports
if roads_df.crs != "EPSG:4326":
    roads_df = roads_df.to_crs("EPSG:4326")

print("Opening GEE flood raster...")
raster_path = "Data/flood/Assam_Sentinel1_Flood_2023.tif"
with rasterio.open(raster_path) as src:
    # Extract coordinates of each road's representative point (midpoint/centroid)
    print("Sampling raster values along road network...")
    points = [Point(geom.interpolate(0.5, normalized=True).x, geom.interpolate(0.5, normalized=True).y) for geom in roads_df.geometry]
    
    # Sample the raster at these points
    coord_list = [(p.x, p.y) for p in points]
    sampled_values = [val[0] for val in src.sample(coord_list)]

# Assign flood risk label based on satellite raster
roads_df['flood_risk'] = [1 if v == 1 else 0 for v in sampled_values]

# Combine with landslide risk using the corrected 'mean_slope' column (> 15 degrees)
SLOPE_THRESHOLD = 15.0
roads_df['landslide_risk'] = 0
if 'mean_slope' in roads_df.columns:
    roads_df.loc[roads_df['mean_slope'] > SLOPE_THRESHOLD, 'landslide_risk'] = 1

# Create a combined final disaster risk score/label for ML
# (1 if either flood or landslide risk is present, 0 otherwise)
roads_df['disaster_risk'] = ((roads_df['flood_risk'] == 1) | (roads_df['landslide_risk'] == 1)).astype(int)

# Save the final labeled dataset
output_path = "Data/processed/assam_roads_final_labeled.gpkg"
roads_df.to_file(output_path, driver="GPKG")

print(f"Success! Final labeled dataset saved to {output_path}")
print(f"Total Flood-Prone Segments: {roads_df['flood_risk'].sum():,}")
print(f"Total Landslide-Prone Segments: {roads_df['landslide_risk'].sum():,}")
print(f"Total High-Risk Segments for ML: {roads_df['disaster_risk'].sum():,}")