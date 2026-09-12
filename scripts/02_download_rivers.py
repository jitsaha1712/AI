import os
import osmnx as ox
import geopandas as gpd

# Increase timeout limit to 5 minutes for large regional queries
ox.settings.timeout = 300

print("Downloading major rivers for Assam from OpenStreetMap...")
# Restrict to only major rivers to drastically reduce query size and avoid timeouts
tags = {'waterway': 'river'}
rivers_gdf = ox.features_from_place("Assam, India", tags=tags)

# Filter for linear or polygonal water geometries
rivers_gdf = rivers_gdf[rivers_gdf.geometry.type.isin(['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'])]

# Save to Data/rivers/
os.makedirs("Data/rivers", exist_ok=True)
output_path = "Data/rivers/assam_rivers.gpkg"
rivers_gdf.to_file(output_path, driver="GPKG")

print(f"Success! Major river network saved to {output_path} with {len(rivers_gdf):,} features.")