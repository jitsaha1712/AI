import os
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text

# Database connection string
DB_URL = "postgresql://postgres:postgres@localhost:5432/assam_routing"
engine = create_engine(DB_URL)


def generate_grid_points(minx, miny, maxx, maxy, grid_size=4):
    """Generates a spatial grid of (lat, lon) points across Assam bounds."""
    lons = np.linspace(minx, maxx, grid_size)
    lats = np.linspace(miny, maxy, grid_size)
    grid = []
    for lat in lats:
        for lon in lons:
            grid.append((lat, lon))
    return grid


def fetch_weather_for_grid(grid_points):
    """Fetches real-time and past rainfall data from Open-Meteo API for spatial grid points."""
    lats = [p[0] for p in grid_points]
    lons = [p[1] for p in grid_points]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat in lats),
        "longitude": ",".join(f"{lon:.4f}" for lon in lons),
        "hourly": "precipitation",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }

    headers = {
        "User-Agent": "AssamSpatialPipeline/1.0 (GIS Research)",
        "Accept": "application/json",
    }

    print("📡 Querying Open-Meteo API for multi-point weather grid...")
    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        data = [data]

    results = []
    for i, loc_data in enumerate(data):
        hourly = loc_data.get("hourly", {})
        precip = hourly.get("precipitation", [])

        # Latest 1-hour rainfall (mm)
        rain_1h = precip[-1] if precip else 0.0

        # Last 24-hour accumulated rainfall (mm)
        rain_24h = sum(precip[-24:]) if len(precip) >= 24 else sum(precip)

        # 7-day cumulative rainfall (mm) - Antecedent Rainfall Index
        ari_7d = sum(precip) if precip else 0.0

        results.append({
            "lat": lats[i],
            "lon": lons[i],
            "rain_1h": float(rain_1h),
            "rain_24h": float(rain_24h),
            "ari_7d": float(ari_7d),
        })

    return pd.DataFrame(results)


def main():
    print("📥 Step 1: Fetching road bounding box from PostGIS...")
    roads_wgs84 = gpd.read_postgis(
        "SELECT edge_id, ST_Transform(geometry, 4326) as geometry FROM road_edges",
        engine,
        geom_col="geometry",
    )
    minx, miny, maxx, maxy = roads_wgs84.total_bounds

    print("🌐 Step 2: Sampling 4x4 spatial weather grid over region...")
    grid_points = generate_grid_points(minx, miny, maxx, maxy, grid_size=4)
    weather_df = fetch_weather_for_grid(grid_points)
    print(
        f"✅ Fetched rainfall data for {len(weather_df)} grid sampling locations."
    )

    print("📍 Step 3: Mapping rainfall values to road edge centroids...")
    weather_gdf = gpd.GeoDataFrame(
        weather_df,
        geometry=gpd.points_from_xy(weather_df.lon, weather_df.lat),
        crs="EPSG:4326",
    )

    roads_wgs84["centroid"] = roads_wgs84.geometry.centroid
    roads_centroids = roads_wgs84.set_geometry("centroid")

    # Nearest neighbor join mapping each road edge to its closest weather sample point
    joined = gpd.sjoin_nearest(roads_centroids, weather_gdf, how="left")

    print("💾 Step 4: Upserting into PostGIS 'dynamic_edge_state' table...")
    now_str = datetime.now().isoformat()
    data_to_update = []

    for _, row in joined.iterrows():
        data_to_update.append({
            "edge_id": int(row["edge_id"]),
            "rain_1h": float(row["rain_1h"]),
            "rain_24h": float(row["rain_24h"]),
            "ari_7d": float(row["ari_7d"]),
            "updated_at": now_str,
        })

    with engine.begin() as conn:
        stmt = text("""
            INSERT INTO dynamic_edge_state (edge_id, rain_1h, rain_24h, ari_7d, updated_at)
            VALUES (:edge_id, :rain_1h, :rain_24h, :ari_7d, :updated_at)
            ON CONFLICT (edge_id) 
            DO UPDATE SET 
                rain_1h = EXCLUDED.rain_1h,
                rain_24h = EXCLUDED.rain_24h,
                ari_7d = EXCLUDED.ari_7d,
                updated_at = EXCLUDED.updated_at;
        """)
        conn.execute(stmt, data_to_update)

    print(
        "🎉 Phase 2 Complete! Dynamic weather state successfully updated in PostGIS."
    )


if __name__ == "__main__":
    main()