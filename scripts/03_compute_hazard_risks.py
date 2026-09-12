import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Database connection string
DB_URL = "postgresql://postgres:postgres@localhost:5432/assam_routing"
engine = create_engine(DB_URL)


def sigmoid(x):
    """Standard sigmoid activation function for smooth probability mapping."""
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def compute_flood_probability(rain_24h, distance_to_river_m, elevation_m):
    """Computes flood probability P(flood) in range [0.0, 1.0]."""
    river_factor = np.exp(-distance_to_river_m / 300.0)  # High within 300m
    elev_factor = np.exp(-elevation_m / 50.0)            # High at low elevations

    logit = (0.05 * rain_24h) + (2.5 * river_factor) + (1.5 * elev_factor) - 3.5
    return sigmoid(logit)


def compute_landslide_probability(slope_deg, rain_1h, ari_7d):
    """Computes landslide probability P(landslide) in range [0.0, 1.0]."""
    slope_factor = np.maximum(0, (slope_deg - 10.0) / 20.0)
    logit = (2.0 * slope_factor) + (0.08 * rain_1h) + (0.02 * ari_7d) - 3.0
    return sigmoid(logit)


def ensure_columns_exist():
    """Ensures hazard result columns exist in dynamic_edge_state table."""
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE dynamic_edge_state 
            ADD COLUMN IF NOT EXISTS p_flood DOUBLE PRECISION DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS p_landslide DOUBLE PRECISION DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS dynamic_weight DOUBLE PRECISION DEFAULT 0.0;
        """))


def main():
    print("🛠️ Step 0: Ensuring schema columns exist in PostGIS...")
    ensure_columns_exist()

    print("📥 Step 1: Loading static and dynamic state from PostGIS...")
    query = """
        SELECT 
            re.edge_id,
            re.length_km,
            re.elevation_m,
            re.slope_deg,
            re.distance_to_river_m,
            des.rain_1h,
            des.rain_24h,
            des.ari_7d
        FROM road_edges re
        JOIN dynamic_edge_state des ON re.edge_id = des.edge_id;
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Loaded features for {len(df)} road edges.")

    print("⚡ Step 2: Computing real-time hazard probabilities...")
    p_flood = compute_flood_probability(
        df["rain_24h"].values,
        df["distance_to_river_m"].values,
        df["elevation_m"].values,
    )

    p_landslide = compute_landslide_probability(
        df["slope_deg"].values,
        df["rain_1h"].values,
        df["ari_7d"].values,
    )

    p_hazard = np.maximum(p_flood, p_landslide)
    dynamic_weight = df["length_km"].values * (1.0 + 10.0 * (p_hazard ** 2))

    df["p_flood"] = np.round(p_flood, 4)
    df["p_landslide"] = np.round(p_landslide, 4)
    df["dynamic_weight"] = np.round(dynamic_weight, 4)

    print("💾 Step 3: Updating hazard scores in 'dynamic_edge_state'...")
    data_to_update = [
        {
            "edge_id": int(row["edge_id"]),
            "p_flood": float(row["p_flood"]),
            "p_landslide": float(row["p_landslide"]),
            "dynamic_weight": float(row["dynamic_weight"]),
        }
        for _, row in df.iterrows()
    ]

    with engine.begin() as conn:
        stmt = text("""
            UPDATE dynamic_edge_state 
            SET 
                p_flood = :p_flood,
                p_landslide = :p_landslide,
                dynamic_weight = :dynamic_weight
            WHERE edge_id = :edge_id;
        """)
        conn.execute(stmt, data_to_update)

    print("🎉 Phase 3 Complete! Real-time hazard risk modeling finished.")


if __name__ == "__main__":
    main()