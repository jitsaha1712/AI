import os
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:postgres@localhost:5432/assam_routing"
engine = create_engine(DB_URL)

os.makedirs("Data/processed", exist_ok=True)


def generate_synthetic_incidents(roads_gdf, n_samples=300):
    """Generates realistic synthetic flood/landslide coordinates across Assam if raw logs aren't present."""
    bounds = roads_gdf.total_bounds
    lons = np.random.uniform(bounds[0], bounds[2], n_samples)
    lats = np.random.uniform(bounds[1], bounds[3], n_samples)
    types = np.random.choice(["flood", "landslide"], size=n_samples, p=[0.6, 0.4])

    df = pd.DataFrame({"latitude": lats, "longitude": lons, "disaster_type": types})
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    ).to_crs(epsg=32645)


def main():
    print("📥 Step 1: Loading road network from PostGIS...")
    roads_gdf = gpd.read_postgis(
        "SELECT edge_id, elevation_m, slope_deg, distance_to_river_m, geometry FROM road_edges",
        engine,
        geom_col="geometry",
    )

    incidents_path = "Data/raw/historical_incidents.csv"
    if os.path.exists(incidents_path):
        print("📍 Loading existing historical incident file...")
        inc_df = pd.read_csv(incidents_path)
        incidents_gdf = gpd.GeoDataFrame(
            inc_df,
            geometry=gpd.points_from_xy(inc_df.longitude, inc_df.latitude),
            crs="EPSG:4326",
        ).to_crs(epsg=32645)
    else:
        print("⚠️ No historical file found. Generating spatial incident dataset...")
        incidents_gdf = generate_synthetic_incidents(roads_gdf)

    print("🔍 Step 2: Performing 100m spatial join with road network...")
    matched = gpd.sjoin_nearest(
        roads_gdf, incidents_gdf, max_distance=100, how="left"
    )

    # Label matching
    matched["is_flood"] = (matched["disaster_type"] == "flood").astype(int)
    matched["is_landslide"] = (matched["disaster_type"] == "landslide").astype(int)

    # Add synthetic weather variations for ML training context
    np.random.seed(42)
    matched["rain_24h_mm"] = np.where(
        matched["is_flood"] == 1,
        np.random.uniform(80, 220, len(matched)),
        np.random.uniform(0, 60, len(matched)),
    )
    matched["rain_1h_mm"] = np.where(
        matched["is_landslide"] == 1,
        np.random.uniform(30, 90, len(matched)),
        np.random.uniform(0, 20, len(matched)),
    )
    matched["ari_7d"] = matched["rain_24h_mm"] * np.random.uniform(1.5, 3.5, len(matched))

    output_path = "Data/processed/training_hazards.csv"
    df_out = pd.DataFrame(matched.drop(columns=["geometry"]))
    df_out.to_csv(output_path, index=False)
    print(f"🎉 Training dataset saved with {len(df_out)} samples to '{output_path}'.")


if __name__ == "__main__":
    main()