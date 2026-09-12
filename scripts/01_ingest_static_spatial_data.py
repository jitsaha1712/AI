import os
import geopandas as gpd
import numpy as np
import rasterio
from sqlalchemy import create_engine, text

# Database connection string
DB_URL = "postgresql://postgres:postgres@localhost:5432/assam_routing"
engine = create_engine(DB_URL)

PROCESSED_DIR = "Data/processed"


def get_best_roads_file():
    priority_files = [
        "assam_roads_final_labeled.gpkg",
        "roads_with_elevation_slope.gpkg",
        "assam_roads_elevation_slope.gpkg",
    ]

    for fname in priority_files:
        full_path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(full_path):
            print(f"📁 Selected road dataset: {full_path}")
            return full_path

    raise FileNotFoundError(
        f"❌ No matching road GeoPackage found in {PROCESSED_DIR}."
    )


def sample_raster_bulk(raster_path, midpoints):
    """High-performance bulk point sampling from a raster file using rasterio.sample."""
    coords = [(pt.x, pt.y) for pt in midpoints]
    with rasterio.open(raster_path) as src:
        samples = src.sample(coords)
        return [
            float(val[0])
            if (val[0] > -9999 and not np.isnan(val[0]))
            else 0.0
            for val in samples
        ]


def main():
    print("📥 Step 1: Loading Road GeoPackage into GeoDataFrame...")
    roads_path = get_best_roads_file()
    gdf_roads = gpd.read_file(roads_path)

    print(f"📊 Loaded {len(gdf_roads)} road edges.")

    # 1. Coordinate Reference System (CRS) check
    if gdf_roads.crs is None or gdf_roads.crs.to_epsg() != 32645:
        print("🔄 Reprojecting roads to EPSG:32645 (UTM Zone 45N)...")
        gdf_roads = gdf_roads.to_crs(epsg=32645)

    # 2. Length calculation in KM
    gdf_roads["length_km"] = gdf_roads.geometry.length / 1000.0

    # 3. Node & Edge ID standardization
    for col in ["u", "source", "source_node", "from"]:
        if col in gdf_roads.columns:
            gdf_roads["source_node"] = gdf_roads[col]
            break
    if "source_node" not in gdf_roads.columns:
        gdf_roads["source_node"] = np.arange(len(gdf_roads))

    for col in ["v", "target", "target_node", "to"]:
        if col in gdf_roads.columns:
            gdf_roads["target_node"] = gdf_roads[col]
            break
    if "target_node" not in gdf_roads.columns:
        gdf_roads["target_node"] = np.arange(len(gdf_roads)) + 1000000

    for col in ["edge_id", "osmid", "id"]:
        if col in gdf_roads.columns:
            gdf_roads["edge_id"] = gdf_roads[col]
            break
    if "edge_id" not in gdf_roads.columns:
        gdf_roads["edge_id"] = np.arange(1, len(gdf_roads) + 1)

    # 4. Compute midpoints once for sampling
    midpoints = [geom.interpolate(0.5, normalized=True) for geom in gdf_roads.geometry]

    # 5. Fast Elevation Sampling
    elev_col = next(
        (c for c in ["elevation_m", "elevation", "elev"] if c in gdf_roads.columns),
        None,
    )
    elev_tif = os.path.join(PROCESSED_DIR, "elevation_utm.tif")

    if elev_col:
        print(f"⛰️ Using existing elevation column: '{elev_col}'")
        gdf_roads["elevation_m"] = gdf_roads[elev_col].astype(float)
    elif os.path.exists(elev_tif):
        print("⛰️ Bulk sampling elevation from elevation_utm.tif (Fast)...")
        gdf_roads["elevation_m"] = sample_raster_bulk(elev_tif, midpoints)
    else:
        gdf_roads["elevation_m"] = 0.0

    # 6. Fast Slope Sampling
    slope_col = next(
        (c for c in ["slope_deg", "slope"] if c in gdf_roads.columns), None
    )
    slope_tif = os.path.join(PROCESSED_DIR, "slope_utm.tif")

    if slope_col:
        print(f"📐 Using existing slope column: '{slope_col}'")
        gdf_roads["slope_deg"] = gdf_roads[slope_col].astype(float)
    elif os.path.exists(slope_tif):
        print("📐 Bulk sampling slope from slope_utm.tif (Fast)...")
        gdf_roads["slope_deg"] = sample_raster_bulk(slope_tif, midpoints)
    else:
        gdf_roads["slope_deg"] = 0.0

    # 7. Distance to river default
    if "distance_to_river_m" not in gdf_roads.columns:
        gdf_roads["distance_to_river_m"] = 9999.0

    # Cleanup list/dict/numpy data types for PostgreSQL compliance
    for c in ["edge_id", "source_node", "target_node"]:
        gdf_roads[c] = gdf_roads[c].apply(
            lambda x: int(x[0])
            if isinstance(x, (list, np.ndarray))
            else int(x)
        )

    required_cols = [
        "edge_id",
        "source_node",
        "target_node",
        "length_km",
        "elevation_m",
        "slope_deg",
        "distance_to_river_m",
        "geometry",
    ]

    gdf_final = gdf_roads[required_cols]

    print("🧹 Step 2: Dropping old PostGIS tables with CASCADE to release constraints...")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS dynamic_edge_state CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS road_edges CASCADE;"))

    print("💾 Step 3: Ingesting into PostGIS 'road_edges' table...")
    gdf_final.to_postgis(
        "road_edges",
        engine,
        if_exists="replace",
        index=False,
        dtype={"geometry": "Geometry(LineString, 32645)"},
    )

    print("⚙️ Step 4: Re-creating PK and 'dynamic_edge_state' table...")
    with engine.begin() as conn:
        # Set Primary Key on road_edges
        conn.execute(text("ALTER TABLE road_edges ADD PRIMARY KEY (edge_id);"))

        # Re-create dynamic_edge_state with Foreign Key
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS dynamic_edge_state (
                edge_id BIGINT PRIMARY KEY REFERENCES road_edges(edge_id) ON DELETE CASCADE,
                rain_1h FLOAT DEFAULT 0.0,
                rain_24h FLOAT DEFAULT 0.0,
                ari_7d FLOAT DEFAULT 0.0,
                visibility_m FLOAT DEFAULT 10000.0,
                wind_speed_ms FLOAT DEFAULT 0.0,
                p_landslide FLOAT DEFAULT 0.0,
                p_flood FLOAT DEFAULT 0.0,
                p_hazard FLOAT DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        )

        # Insert corresponding dynamic states
        conn.execute(
            text("""
            INSERT INTO dynamic_edge_state (edge_id)
            SELECT edge_id FROM road_edges
            ON CONFLICT (edge_id) DO NOTHING;
        """)
        )

    print("✅ Phase 1 Complete! Road network successfully loaded into PostGIS.")


if __name__ == "__main__":
    main()