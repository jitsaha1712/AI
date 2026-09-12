from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:postgres@localhost:5432/assam_routing"
engine = create_engine(DB_URL)

sql = text("""
ALTER TABLE dynamic_edge_state 
ADD COLUMN IF NOT EXISTS rain_1h_mm DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS rain_24h_mm DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS ari_7d DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS wind_speed_kmh DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS visibility_km DOUBLE PRECISION DEFAULT 10.0,
ADD COLUMN IF NOT EXISTS p_landslide DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS p_flood DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS p_hazard DOUBLE PRECISION DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS shap_explanation TEXT DEFAULT 'Normal Conditions';
""")

with engine.begin() as conn:
    conn.execute(sql)

print("✅ Schema updated successfully!")
