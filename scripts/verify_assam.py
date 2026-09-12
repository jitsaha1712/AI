import geopandas as gpd

# Load the filtered Assam file
file_path = "Data/processed/assam_roads_elevation_slope.gpkg"
print(f"Loading filtered Assam road network from: {file_path}")
roads_df = gpd.read_file(file_path)

# 1. Check total number of records remaining
print(f"\nTotal valid Assam road segments: {len(roads_df):,}")

# 2. Preview elevation and slope values for the first 5 rows
print("\nSample Values (First 5 segments):")
print(roads_df[["osm_id", "fclass", "mean_elevation", "max_elevation", "mean_slope", "max_slope"]].head())

# 3. Double-check for any remaining missing (NaN) values in key columns
print("\nMissing Values Count in Filtered Dataset:")
print(roads_df[["mean_elevation", "mean_slope"]].isnull().sum())