import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import psycopg2
from datetime import timedelta


# ============================================================
# CONFIGURATION
# ============================================================

RAINFALL_DIR = r"C:\NERProject\Data\historical_rainfall"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "assam_routing",
    "user": "postgres",
    "password": "postgres"
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_rainfall_for_location(ds, longitude, latitude):
    """
    Get rainfall from the nearest IMERG grid cell.

    IMERG file dimensions are:
        precipitation(time, lon, lat)
    """

    rain = ds["precipitation"].isel(time=0)

    # Find nearest longitude and latitude
    rain_value = rain.sel(
        lon=longitude,
        lat=latitude,
        method="nearest"
    ).item()

    # Handle missing / invalid values
    if not np.isfinite(rain_value):
        return 0.0

    return float(rain_value)


# ============================================================
# READ LANDSLIDE EVENTS
# ============================================================

print("Connecting to PostgreSQL...")

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    event_id,
    event_date,
    longitude,
    latitude,
    location_accuracy,
    landslide_trigger,
    landslide_size
FROM historical_landslides
WHERE event_date IS NOT NULL
  AND longitude IS NOT NULL
  AND latitude IS NOT NULL
ORDER BY event_date, event_id;
"""

events = pd.read_sql(query, conn)

print(f"Landslide events found: {len(events)}")


# ============================================================
# FIND ALL RAINFALL FILES
# ============================================================

print("\nReading historical rainfall files...")

rainfall_files = glob.glob(
    os.path.join(RAINFALL_DIR, "IMERG_*.nc4")
)

print(f"Rainfall files found: {len(rainfall_files)}")


# ============================================================
# CREATE DATE → FILE LOOKUP
# ============================================================

rainfall_by_date = {}

for file_path in rainfall_files:

    filename = os.path.basename(file_path)

    # Example:
    # IMERG_20070709.nc4

    date_text = filename.replace(
        "IMERG_", ""
    ).replace(
        ".nc4", ""
    )

    try:
        date_value = pd.to_datetime(
            date_text
        ).date()

        rainfall_by_date[date_value] = file_path

    except Exception:
        print(
            f"Skipping invalid filename: {filename}"
        )


print(
    f"Usable rainfall dates: "
    f"{len(rainfall_by_date)}"
)


# ============================================================
# PROCESS EACH LANDSLIDE
# ============================================================

results = []

total_events = len(events)

print("\nExtracting rainfall features...")
print("========================================")


for index, event in events.iterrows():

    event_id = int(event["event_id"])
    event_date = event["event_date"]

    longitude = float(event["longitude"])
    latitude = float(event["latitude"])

    print(
        f"[{index + 1}/{total_events}] "
        f"Event {event_id} "
        f"({event_date})"
    )

    # --------------------------------------------------------
    # Daily rainfall dictionary
    # --------------------------------------------------------

    daily_rainfall = {}

    # Event day + previous 10 days
    for days_before in range(0, 11):

        current_date = (
            event_date -
            timedelta(days=days_before)
        )

        file_path = rainfall_by_date.get(
            current_date
        )

        if file_path is None:

            print(
                f"    WARNING: rainfall missing "
                f"for {current_date}"
            )

            daily_rainfall[current_date] = np.nan

            continue

        try:

            with xr.open_dataset(file_path) as ds:

                rain_value = get_rainfall_for_location(
                    ds,
                    longitude,
                    latitude
                )

                daily_rainfall[current_date] = (
                    rain_value
                )

        except Exception as e:

            print(
                f"    ERROR reading "
                f"{current_date}: {e}"
            )

            daily_rainfall[current_date] = np.nan


    # --------------------------------------------------------
    # Convert to ordered rainfall list
    #
    # [event day, 1 day before, ..., 10 days before]
    # --------------------------------------------------------

    rain_values = np.array([
        daily_rainfall.get(
            event_date - timedelta(days=i),
            np.nan
        )
        for i in range(0, 11)
    ], dtype=float)


    # --------------------------------------------------------
    # Replace invalid values with zero ONLY for
    # aggregation after recording missing information
    # --------------------------------------------------------

    valid_values = rain_values[
        np.isfinite(rain_values)
    ]

    if len(valid_values) == 0:

        print(
            "    WARNING: no valid rainfall data"
        )

        rain_values = np.zeros(11)

    else:

        rain_values = np.nan_to_num(
            rain_values,
            nan=0.0
        )


    # --------------------------------------------------------
    # Rainfall features
    # --------------------------------------------------------

    rain_event_day = rain_values[0]

    rain_1d = (
        rain_values[0]
    )

    rain_3d = (
        rain_values[0:3].sum()
    )

    rain_7d = (
        rain_values[0:7].sum()
    )

    rain_10d = (
        rain_values[0:10].sum()
    )

    max_rain_3d = 0.0

    for i in range(0, 9):

        three_day_total = (
            rain_values[i:i + 3].sum()
        )

        max_rain_3d = max(
            max_rain_3d,
            three_day_total
        )


    max_rain_7d = 0.0

    for i in range(0, 5):

        seven_day_total = (
            rain_values[i:i + 7].sum()
        )

        max_rain_7d = max(
            max_rain_7d,
            seven_day_total
        )


    # --------------------------------------------------------
    # Heavy rainfall days
    #
    # Initial threshold:
    # >= 50 mm/day
    #
    # This is a feature threshold, NOT the final
    # landslide threshold.
    # --------------------------------------------------------

    heavy_rain_days = int(
        np.sum(rain_values >= 50.0)
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "event_id": event_id,

        "event_date": event_date,

        "longitude": longitude,

        "latitude": latitude,

        "location_accuracy":
            event["location_accuracy"],

        "landslide_trigger":
            event["landslide_trigger"],

        "landslide_size":
            event["landslide_size"],

        "rain_event_day":
            rain_event_day,

        "rain_1d":
            rain_1d,

        "rain_3d":
            rain_3d,

        "rain_7d":
            rain_7d,

        "rain_10d":
            rain_10d,

        "max_rain_3d":
            max_rain_3d,

        "max_rain_7d":
            max_rain_7d,

        "heavy_rain_days":
            heavy_rain_days,

        "valid_rainfall_days":
            len(valid_values)
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

result_df = pd.DataFrame(results)


# ============================================================
# SAVE CSV
# ============================================================

output_csv = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_rainfall_features.csv"
)

result_df.to_csv(
    output_csv,
    index=False
)


print("\n========================================")
print("Rainfall extraction completed!")
print("========================================")
print(
    f"Landslide events processed: "
    f"{len(result_df)}"
)
print(
    f"Output CSV: {output_csv}"
)
print("========================================")


# ============================================================
# CREATE POSTGRESQL TABLE
# ============================================================

print("\nCreating PostgreSQL table...")

cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS
historical_landslide_rainfall_features;
""")


cursor.execute("""
CREATE TABLE
historical_landslide_rainfall_features (

    event_id bigint PRIMARY KEY,

    event_date date,

    longitude double precision,

    latitude double precision,

    location_accuracy text,

    landslide_trigger text,

    landslide_size text,

    rain_event_day double precision,

    rain_1d double precision,

    rain_3d double precision,

    rain_7d double precision,

    rain_10d double precision,

    max_rain_3d double precision,

    max_rain_7d double precision,

    heavy_rain_days integer,

    valid_rainfall_days integer
);
""")


# ============================================================
# INSERT RESULTS
# ============================================================

insert_query = """
INSERT INTO
historical_landslide_rainfall_features (

    event_id,
    event_date,
    longitude,
    latitude,
    location_accuracy,
    landslide_trigger,
    landslide_size,
    rain_event_day,
    rain_1d,
    rain_3d,
    rain_7d,
    rain_10d,
    max_rain_3d,
    max_rain_7d,
    heavy_rain_days,
    valid_rainfall_days

)
VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s
);
"""


for _, row in result_df.iterrows():

    cursor.execute(
        insert_query,
        (
            int(row["event_id"]),
            row["event_date"],
            float(row["longitude"]),
            float(row["latitude"]),
            row["location_accuracy"],
            row["landslide_trigger"],
            row["landslide_size"],
            float(row["rain_event_day"]),
            float(row["rain_1d"]),
            float(row["rain_3d"]),
            float(row["rain_7d"]),
            float(row["rain_10d"]),
            float(row["max_rain_3d"]),
            float(row["max_rain_7d"]),
            int(row["heavy_rain_days"]),
            int(row["valid_rainfall_days"])
        )
    )


conn.commit()

cursor.close()
conn.close()


print("\n========================================")
print("PostgreSQL import completed!")
print("========================================")
print(
    "Table: "
    "historical_landslide_rainfall_features"
)
print(
    f"Records inserted: {len(result_df)}"
)
print("========================================")