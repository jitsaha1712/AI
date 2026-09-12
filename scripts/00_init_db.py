import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text

# Update credentials if your PostgreSQL password isn't 'postgres'
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "assam_routing"


def create_database_if_not_exists():
    print("🔌 Connecting to default PostgreSQL instance...")
    try:
        # Connect to default 'postgres' db with autocommit to create new DB
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if assam_routing exists
        cursor.execute(
            f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'"
        )
        exists = cursor.fetchone()

        if not exists:
            print(f"🛠️ Creating database '{DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE {DB_NAME};")
            print(f"✅ Database '{DB_NAME}' created successfully.")
        else:
            print(f"ℹ️ Database '{DB_NAME}' already exists.")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        print(
            "💡 Tip: Ensure PostgreSQL service is running and check your password."
        )
        exit(1)


def init_postgis_and_tables():
    target_db_url = (
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(target_db_url)

    sql_setup = """
    -- 1. Enable PostGIS Extension
    CREATE EXTENSION IF NOT EXISTS postgis;

    -- 2. Drop existing tables if rebuilding
    DROP TABLE IF EXISTS dynamic_edge_state CASCADE;
    DROP TABLE IF EXISTS road_edges CASCADE;

    -- 3. Static Road Network & Terrain Table
    CREATE TABLE road_edges (
        edge_id BIGINT PRIMARY KEY,
        source_node BIGINT NOT NULL,
        target_node BIGINT NOT NULL,
        length_km FLOAT NOT NULL,
        elevation_m FLOAT DEFAULT 0.0,
        slope_deg FLOAT DEFAULT 0.0,
        distance_to_river_m FLOAT DEFAULT 9999.0,
        historical_landslide_freq INT DEFAULT 0,
        historical_flood_freq INT DEFAULT 0,
        geom GEOMETRY(LineString, 32645) -- EPSG:32645 (UTM Zone 45N)
    );

    CREATE INDEX idx_road_edges_geom ON road_edges USING GIST(geom);

    -- 4. Dynamic Weather & ML Risk Table
    CREATE TABLE dynamic_edge_state (
        edge_id BIGINT PRIMARY KEY REFERENCES road_edges(edge_id) ON DELETE CASCADE,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        rain_1h_mm FLOAT DEFAULT 0.0,
        rain_24h_mm FLOAT DEFAULT 0.0,
        ari_7d FLOAT DEFAULT 0.0,
        rain_zscore FLOAT DEFAULT 0.0,
        visibility_km FLOAT DEFAULT 10.0,
        wind_speed_kmh FLOAT DEFAULT 0.0,
        p_landslide FLOAT DEFAULT 0.0,
        p_flood FLOAT DEFAULT 0.0,
        p_hazard FLOAT DEFAULT 0.0,
        shap_explanation TEXT
    );

    CREATE INDEX idx_dynamic_updated_at ON dynamic_edge_state(updated_at);
    """

    print(
        f"⚙️ Enabling PostGIS extension and creating tables in '{DB_NAME}'..."
    )
    with engine.begin() as conn:
        conn.execute(text(sql_setup))
    print(
        "✅ PostGIS enabled and tables ('road_edges', 'dynamic_edge_state') created successfully!"
    )


if __name__ == "__main__":
    create_database_if_not_exists()
    init_postgis_and_tables()