import pandas as pd
import psycopg2
from pyproj import Transformer


# ==========================================
# 1. File path
# ==========================================

CSV_PATH = r"C:\NERProject\Data\landslide\assam_landslides.csv"


# ==========================================
# 2. PostgreSQL connection
# ==========================================

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="assam_routing",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()


# ==========================================
# 3. Read CSV
# ==========================================

print("Reading landslide CSV...")

df = pd.read_csv(CSV_PATH)

print("Rows found:", len(df))


# ==========================================
# 4. Coordinate transformation
# WGS84 -> UTM Zone 45N
# ==========================================

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32645",
    always_xy=True
)


# ==========================================
# 5. Insert records
# ==========================================

insert_query = """
INSERT INTO historical_landslides (
    event_id,
    event_date,
    event_time,
    event_title,
    event_description,
    location_description,
    location_accuracy,
    landslide_category,
    landslide_trigger,
    landslide_size,
    landslide_setting,
    fatality_count,
    injury_count,
    country_name,
    country_code,
    admin_division_name,
    gazeteer_closest_point,
    gazeteer_distance,
    longitude,
    latitude,
    geometry
)
VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    ST_Transform(
        ST_SetSRID(
            ST_MakePoint(%s, %s),
            4326
        ),
        32645
    )
)
ON CONFLICT (event_id) DO NOTHING;
"""


# ==========================================
# 6. Process each landslide
# ==========================================

inserted = 0

for _, row in df.iterrows():

    try:

        # ------------------------------
        # Date
        # ------------------------------

        event_date = pd.to_datetime(
            row["event_date"],
            errors="coerce"
        )

        if pd.isna(event_date):
            event_date = None
        else:
            event_date = event_date.date()


        # ------------------------------
        # Coordinates
        # ------------------------------

        latitude = float(row["latitude"])
        longitude = float(row["longitude"])


        # Validate Assam coordinate range
        if not (24 <= latitude <= 28):
            print("Skipping invalid latitude:", latitude)
            continue

        if not (89 <= longitude <= 96):
            print("Skipping invalid longitude:", longitude)
            continue


        # ------------------------------
        # Missing numeric values
        # ------------------------------

        fatality_count = (
            None
            if pd.isna(row["fatality_count"])
            else int(row["fatality_count"])
        )

        injury_count = (
            None
            if pd.isna(row["injury_count"])
            else int(row["injury_count"])
        )

        gazeteer_distance = (
            None
            if pd.isna(row["gazeteer_distance"])
            else float(row["gazeteer_distance"])
        )


        # ------------------------------
        # Insert
        # ------------------------------

        cur.execute(
            insert_query,
            (
                int(row["event_id"]),
                event_date,
                None if pd.isna(row["event_time"]) else str(row["event_time"]),

                None if pd.isna(row["event_title"]) else str(row["event_title"]),
                None if pd.isna(row["event_description"]) else str(row["event_description"]),
                None if pd.isna(row["location_description"]) else str(row["location_description"]),

                None if pd.isna(row["location_accuracy"]) else str(row["location_accuracy"]),
                None if pd.isna(row["landslide_category"]) else str(row["landslide_category"]),
                None if pd.isna(row["landslide_trigger"]) else str(row["landslide_trigger"]),
                None if pd.isna(row["landslide_size"]) else str(row["landslide_size"]),
                None if pd.isna(row["landslide_setting"]) else str(row["landslide_setting"]),

                fatality_count,
                injury_count,

                None if pd.isna(row["country_name"]) else str(row["country_name"]),
                None if pd.isna(row["country_code"]) else str(row["country_code"]),
                None if pd.isna(row["admin_division_name"]) else str(row["admin_division_name"]),

                None if pd.isna(row["gazeteer_closest_point"])
                else str(row["gazeteer_closest_point"]),

                gazeteer_distance,

                longitude,
                latitude,

                longitude,
                latitude
            )
        )

        inserted += 1

    except Exception as e:

        print(
            f"Error inserting event {row.get('event_id')}: {e}"
        )


# ==========================================
# 7. Save
# ==========================================

conn.commit()

print()
print("====================================")
print("Landslide import completed!")
print("Records inserted:", inserted)
print("====================================")


# ==========================================
# 8. Close
# ==========================================

cur.close()
conn.close()