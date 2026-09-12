import os
import geopandas as gpd
import requests
from shapely.geometry import LineString
from sqlalchemy import create_engine, text

# Database connection string
DB_URL = "postgresql://postgres:postgres@localhost:5432/assam_routing"
engine = create_engine(DB_URL)


def fetch_osm_rivers(minx, miny, maxx, maxy):
    """Fetches major river network from Overpass API using proper HTTP headers and active mirrors."""
    # Focus on rivers and canals to prevent 504 Gateway Timeout over large bounding box
    overpass_query = f"""
    [out:json][timeout:180];
    (
      way["waterway"="river"]({miny},{minx},{maxy},{maxx});
      way["waterway"="canal"]({miny},{minx},{maxy},{maxx});
    );
    out geom;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]

    headers = {
        "User-Agent": "AssamSpatialPipeline/1.0 (GIS Research)",
        "Accept": "application/json, */*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    for url in endpoints:
        print(f"📡 Querying Overpass server: {url}...")
        try:
            response = requests.post(
                url,
                data={"data": overpass_query},
                headers=headers,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            print(f"⚠️ Failed on {url}: {e}. Trying fallback server...")

    raise RuntimeError(
        "❌ All Overpass API endpoints failed. Check network connection or try again shortly."
    )


def main():
    print("📥 Step 1: Fetching road bounding box from PostGIS...")
    roads_wgs84 = gpd.read_postgis(
        "SELECT edge_id, ST_Transform(geometry, 4326) as geometry FROM road_edges",
        engine,
        geom_col="geometry",
    )
    minx, miny, maxx, maxy = roads_wgs84.total_bounds
    print(
        f"📍 Bounding Box (WGS84): Min({minx:.3f}, {miny:.3f}) -> Max({maxx:.3f}, {maxy:.3f})"
    )

    print(
        "🌊 Step 2: Downloading major river network from OpenStreetMap..."
    )
    data = fetch_osm_rivers(minx, miny, maxx, maxy)

    geoms = []
    for element in data.get("elements", []):
        if "geometry" in element:
            coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
            if len(coords) >= 2:
                geoms.append(LineString(coords))

    if not geoms:
        print(
            "⚠️ No river polylines returned for this area. Keeping default distances."
        )
        return

    # Convert downloaded river polylines into GeoDataFrame and reproject to UTM 45N
    gdf_rivers = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326").to_crs(
        epsg=32645
    )
    print(f"✅ Successfully downloaded {len(gdf_rivers)} river segments!")

    # Save locally in Data/raw/
    os.makedirs("Data/raw", exist_ok=True)
    river_output_path = "Data/raw/assam_rivers_osm.geojson"
    gdf_rivers.to_file(river_output_path, driver="GeoJSON")
    print(f"💾 Saved river dataset locally to '{river_output_path}'")

    print("📏 Step 3: Calculating distance from each road segment to nearest river...")
    roads_utm = gpd.read_postgis(
        "SELECT edge_id, geometry FROM road_edges", engine, geom_col="geometry"
    )
    river_union = gdf_rivers.geometry.union_all()

    # Metric distance calculation in meters (UTM Zone 45N projection)
    roads_utm["distance_to_river_m"] = [
        float(geom.distance(river_union)) for geom in roads_utm.geometry
    ]

    print("💾 Step 4: Updating PostGIS 'road_edges' table...")
    with engine.begin() as conn:
        data_to_update = [
            {
                "dist": float(row["distance_to_river_m"]),
                "id": int(row["edge_id"]),
            }
            for _, row in roads_utm.iterrows()
        ]
        stmt = text(
            "UPDATE road_edges SET distance_to_river_m = :dist WHERE edge_id = :id"
        )
        conn.execute(stmt, data_to_update)

    print(
        "🎉 Success! PostGIS 'road_edges' table updated with river distances."
    )


if __name__ == "__main__":
    main()