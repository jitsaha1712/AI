import os
import time
import requests
import numpy as np
import pandas as pd
import psycopg2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\NERProject"

WEATHER_GRID_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "live_weather_grid.csv"
)

OUTPUT_WEATHER_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "live_weather_grid.csv"
)

OUTPUT_ROAD_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "live_road_features.csv"
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "assam_routing",
    "user": "postgres",
    "password": "postgres",  
}


# ============================================================
# OPEN-METEO API
# ============================================================

API_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# STEP 1
# READ EXISTING WEATHER GRID
# ============================================================

print("\n==========================================")
print("STEP 1: Reading weather grid")
print("==========================================")

if not os.path.exists(WEATHER_GRID_FILE):
    raise FileNotFoundError(
        f"Weather grid file not found:\n{WEATHER_GRID_FILE}"
    )

weather_grid = pd.read_csv(
    WEATHER_GRID_FILE
)

required_grid_columns = [
    "grid_id",
    "longitude",
    "latitude"
]

for col in required_grid_columns:
    if col not in weather_grid.columns:
        raise ValueError(
            f"Missing required column in weather grid: {col}"
        )


# Remove duplicate grid IDs

weather_grid = weather_grid.drop_duplicates(
    subset=["grid_id"]
).reset_index(drop=True)

print(
    f"Weather grid points: {len(weather_grid)}"
)


# ============================================================
# STEP 2
# FETCH 10 DAYS OF RAINFALL
# ============================================================

print("\n==========================================")
print("STEP 2: Fetching live weather + 10-day rain")
print("==========================================")

all_weather_rows = []

# Number of coordinates per API request
CHUNK_SIZE = 50


for start in range(
    0,
    len(weather_grid),
    CHUNK_SIZE
):

    end = min(
        start + CHUNK_SIZE,
        len(weather_grid)
    )

    chunk = weather_grid.iloc[
        start:end
    ].copy()

    print(
        f"Fetching points {start + 1} to {end}..."
    )


    # --------------------------------------------------------
    # Prepare coordinates
    # --------------------------------------------------------

    latitudes = ",".join(
        str(float(x))
        for x in chunk["latitude"]
    )

    longitudes = ",".join(
        str(float(x))
        for x in chunk["longitude"]
    )


    # --------------------------------------------------------
    # Open-Meteo parameters
    # --------------------------------------------------------

    params = {

        "latitude": latitudes,
        "longitude": longitudes,

        "current": (
            "precipitation,"
            "temperature_2m,"
            "visibility,"
            "wind_speed_10m,"
            "wind_gusts_10m"
        ),

        "hourly": "precipitation",

        "past_days": 10,

        "forecast_days": 1,

        "timezone": "Asia/Kolkata"
    }


    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        print(
            "First API request failed:"
        )

        print(e)

        print(
            "Retrying in 3 seconds..."
        )

        time.sleep(3)

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()


    # --------------------------------------------------------
    # Normalize response
    # --------------------------------------------------------

    if isinstance(data, dict):
        data = [data]


    if len(data) != len(chunk):

        raise RuntimeError(
            f"Expected {len(chunk)} weather responses "
            f"but received {len(data)}"
        )


    # ========================================================
    # PROCESS EACH WEATHER GRID POINT
    # ========================================================

    for local_index, weather in enumerate(data):

        grid_row = chunk.iloc[local_index]


        # ----------------------------------------------------
        # Current weather
        # ----------------------------------------------------

        current = weather.get(
            "current",
            {}
        )


        # ----------------------------------------------------
        # Hourly rainfall
        # ----------------------------------------------------

        hourly = weather.get(
            "hourly",
            {}
        )

        times = hourly.get(
            "time",
            []
        )

        rain_values = hourly.get(
            "precipitation",
            []
        )


        # ----------------------------------------------------
        # Check rainfall data
        # ----------------------------------------------------

        if len(times) != len(rain_values):

            raise RuntimeError(
                "Hourly time and precipitation "
                "lengths do not match"
            )


        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # Open-Meteo may return precipitation as a
        # NumPy array.
        #
        # Therefore we convert it safely into NumPy,
        # remove NaN values and then create DataFrame.
        # ----------------------------------------------------

        rain_values = np.asarray(
            rain_values,
            dtype=float
        )

        rain_values = np.nan_to_num(
            rain_values,
            nan=0.0
        )


        # ----------------------------------------------------
        # Create hourly dataframe
        # ----------------------------------------------------

        hourly_df = pd.DataFrame({

            "time": pd.to_datetime(
                times
            ),

            "precipitation": rain_values

        })


        # ----------------------------------------------------
        # Create date column
        # ----------------------------------------------------

        hourly_df["date"] = (
            hourly_df["time"].dt.date
        )


        # ----------------------------------------------------
        # Convert hourly rainfall to daily rainfall
        # ----------------------------------------------------

        daily_rain = (

            hourly_df
            .groupby("date")["precipitation"]
            .sum()
            .reset_index()
        )


        # ----------------------------------------------------
        # Last 10 available days
        # ----------------------------------------------------

        last_10 = daily_rain.tail(
            10
        )


        # ----------------------------------------------------
        # Last 7 days
        # ----------------------------------------------------

        last_7 = daily_rain.tail(
            7
        )


        # ----------------------------------------------------
        # Last 3 days
        # ----------------------------------------------------

        last_3 = daily_rain.tail(
            3
        )


        # ====================================================
        # RAINFALL FEATURES
        # ====================================================

        # Total rainfall over last 10 days

        rain_10d = float(
            last_10["precipitation"].sum()
        )


        # Total rainfall over last 7 days

        rain_7d = float(
            last_7["precipitation"].sum()
        )


        # Total rainfall over last 3 days

        rain_3d = float(
            last_3["precipitation"].sum()
        )


        # Maximum rainfall in any day
        # during the last 3 days

        if len(last_3) > 0:

            max_rain_3d = float(
                last_3["precipitation"].max()
            )

        else:

            max_rain_3d = 0.0


        # Maximum rainfall in any day
        # during the last 7 days

        if len(last_7) > 0:

            max_rain_7d = float(
                last_7["precipitation"].max()
            )

        else:

            max_rain_7d = 0.0


        # ----------------------------------------------------
        # Heavy rainfall days
        #
        # Threshold = 25 mm/day
        # ----------------------------------------------------

        heavy_rain_days = int(
            (
                last_10["precipitation"] >= 25.0
            ).sum()
        )


        # ====================================================
        # CURRENT WEATHER FEATURES
        # ====================================================

        current_rain = float(
            current.get(
                "precipitation",
                0.0
            ) or 0.0
        )


        temperature = current.get(
            "temperature_2m",
            np.nan
        )

        visibility = current.get(
            "visibility",
            np.nan
        )

        wind = current.get(
            "wind_speed_10m",
            np.nan
        )

        gust = current.get(
            "wind_gusts_10m",
            np.nan
        )


        # Convert safely to float

        try:
            temperature = float(
                temperature
            )
        except:
            temperature = np.nan


        try:
            visibility = float(
                visibility
            )
        except:
            visibility = np.nan


        try:
            wind = float(
                wind
            )
        except:
            wind = np.nan


        try:
            gust = float(
                gust
            )
        except:
            gust = np.nan


        # ====================================================
        # STORE GRID RESULT
        # ====================================================

        all_weather_rows.append({

            "grid_id": int(
                grid_row["grid_id"]
            ),

            "longitude": float(
                grid_row["longitude"]
            ),

            "latitude": float(
                grid_row["latitude"]
            ),

            "current_rain_mm": current_rain,

            "rain_24h_mm": float(
                last_10[
                    "precipitation"
                ].tail(1).sum()
            ),

            "rain_3d_mm": rain_3d,

            "rain_7d_mm": rain_7d,

            "rain_10d_mm": rain_10d,

            "max_rain_3d_mm": max_rain_3d,

            "max_rain_7d_mm": max_rain_7d,

            "heavy_rain_days": heavy_rain_days,

            "current_temperature_c": temperature,

            "current_visibility_m": visibility,

            "current_wind_kmh": wind,

            "current_wind_gust_kmh": gust
        })


    # Small delay between requests

    time.sleep(0.5)


# ============================================================
# STEP 3
# SAVE UPDATED WEATHER GRID
# ============================================================

print("\n==========================================")
print("STEP 3: Saving updated weather grid")
print("==========================================")

live_weather = pd.DataFrame(
    all_weather_rows
)


if len(live_weather) == 0:

    raise RuntimeError(
        "No weather data was returned."
    )


live_weather.to_csv(
    OUTPUT_WEATHER_FILE,
    index=False
)


print(
    f"Saved: {OUTPUT_WEATHER_FILE}"
)

print(
    f"Rows: {len(live_weather)}"
)


# ============================================================
# RAINFALL SUMMARY
# ============================================================

print("\nRainfall summary:")

print(
    live_weather[
        [
            "current_rain_mm",
            "rain_24h_mm",
            "rain_3d_mm",
            "rain_7d_mm",
            "rain_10d_mm",
            "max_rain_3d_mm",
            "max_rain_7d_mm"
        ]
    ].describe()
)


# ============================================================
# STEP 4
# LOAD ROUTING ROADS
# ============================================================

print("\n==========================================")
print("STEP 4: Reading routing roads")
print("==========================================")

try:

    conn = psycopg2.connect(
        **DB_CONFIG
    )

except Exception as e:

    raise RuntimeError(
        f"Could not connect to PostgreSQL:\n{e}"
    )


sql = """
SELECT
    edge_id,

    ST_X(
        ST_Transform(
            ST_Centroid(geometry),
            4326
        )
    ) AS longitude,

    ST_Y(
        ST_Transform(
            ST_Centroid(geometry),
            4326
        )
    ) AS latitude,

    length_km,

    elevation_m,

    slope_deg,

    distance_to_river_m

FROM routing_noded

WHERE geometry IS NOT NULL

ORDER BY edge_id;
"""


try:

    roads = pd.read_sql(
        sql,
        conn
    )

finally:

    conn.close()


print(
    f"Routing roads: {len(roads)}"
)


if len(roads) == 0:

    raise RuntimeError(
        "No routing roads were returned."
    )


# ============================================================
# STEP 5
# MAP EACH ROAD TO NEAREST WEATHER CELL
# ============================================================

print("\n==========================================")
print("STEP 5: Mapping roads to nearest weather")
print("==========================================")


# ------------------------------------------------------------
# Weather coordinates
# ------------------------------------------------------------

grid_coords = live_weather[
    [
        "longitude",
        "latitude"
    ]
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# Road coordinates
# ------------------------------------------------------------

road_coords = roads[
    [
        "longitude",
        "latitude"
    ]
].to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# Array for nearest weather grid index
# ------------------------------------------------------------

nearest_grid_index = np.empty(
    len(roads),
    dtype=np.int32
)


# ------------------------------------------------------------
# Process roads in chunks
# ------------------------------------------------------------

ROAD_CHUNK_SIZE = 10000


for start in range(
    0,
    len(roads),
    ROAD_CHUNK_SIZE
):

    end = min(
        start + ROAD_CHUNK_SIZE,
        len(roads)
    )


    road_chunk = road_coords[
        start:end
    ]


    # --------------------------------------------------------
    # Difference between every road and every grid cell
    # --------------------------------------------------------

    diff = (

        road_chunk[:, None, :]
        -
        grid_coords[None, :, :]

    )


    # --------------------------------------------------------
    # Squared Euclidean distance
    # --------------------------------------------------------

    dist_sq = np.sum(
        diff * diff,
        axis=2
    )


    # --------------------------------------------------------
    # Find nearest weather cell
    # --------------------------------------------------------

    nearest_grid_index[
        start:end
    ] = np.argmin(
        dist_sq,
        axis=1
    )


    print(
        f"Mapped roads {start + 1} to {end}"
    )


roads["grid_index"] = (
    nearest_grid_index
)


# ============================================================
# STEP 6
# JOIN WEATHER FEATURES TO ROADS
# ============================================================

print("\n==========================================")
print("STEP 6: Joining live weather features")
print("==========================================")


weather_lookup = (
    live_weather
    .reset_index(drop=True)
    .copy()
)


weather_lookup[
    "grid_index"
] = weather_lookup.index


road_features = roads.merge(

    weather_lookup[
        [

            "grid_index",

            "grid_id",

            "current_rain_mm",

            "rain_24h_mm",

            "rain_3d_mm",

            "rain_7d_mm",

            "rain_10d_mm",

            "max_rain_3d_mm",

            "max_rain_7d_mm",

            "heavy_rain_days",

            "current_temperature_c",

            "current_visibility_m",

            "current_wind_kmh",

            "current_wind_gust_kmh"

        ]
    ],

    on="grid_index",

    how="left"
)


# ============================================================
# STEP 7
# CREATE LANDSLIDE MODEL FEATURES
# ============================================================

print("\n==========================================")
print("STEP 7: Creating model-ready features")
print("==========================================")


# ------------------------------------------------------------
# Landslide model
# ------------------------------------------------------------

road_features[
    "rain_event_day"
] = road_features[
    "current_rain_mm"
]


road_features[
    "rain_3d"
] = road_features[
    "rain_3d_mm"
]


road_features[
    "rain_7d"
] = road_features[
    "rain_7d_mm"
]


road_features[
    "rain_10d"
] = road_features[
    "rain_10d_mm"
]


road_features[
    "max_rain_3d"
] = road_features[
    "max_rain_3d_mm"
]


road_features[
    "max_rain_7d"
] = road_features[
    "max_rain_7d_mm"
]


# ============================================================
# STEP 8
# CREATE FLOOD MODEL FEATURES
# ============================================================

# The trained flood model currently contains feature names
# ending with "_2023".
#
# We are NOT using flood_risk as an input here because that
# would leak the training label.
#
# Instead, live weather values are mapped to the same
# rainfall feature meanings.


road_features[
    "rain_1d_live"
] = road_features[
    "rain_24h_mm"
]


road_features[
    "rain_3d_live"
] = road_features[
    "rain_3d_mm"
]


road_features[
    "rain_7d_live"
] = road_features[
    "rain_7d_mm"
]


road_features[
    "max_rain_3d_live"
] = road_features[
    "max_rain_3d_mm"
]


road_features[
    "max_rain_7d_live"
] = road_features[
    "max_rain_7d_mm"
]


# ============================================================
# STEP 9
# VALIDATE DATA
# ============================================================

print("\n==========================================")
print("STEP 8: Validation")
print("==========================================")


required_features = [

    "rain_event_day",

    "rain_3d",

    "rain_7d",

    "rain_10d",

    "max_rain_3d",

    "max_rain_7d",

    "heavy_rain_days",

    "elevation_m",

    "slope_deg",

    "distance_to_river_m",

    "rain_1d_live",

    "rain_3d_live",

    "rain_7d_live",

    "max_rain_3d_live",

    "max_rain_7d_live"
]


# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------

for col in required_features:

    if col not in road_features.columns:

        raise ValueError(
            f"Missing required feature: {col}"
        )


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing_values = (
    road_features[
        required_features
    ]
    .isna()
    .sum()
)


print(
    "\nMissing values:"
)

print(
    missing_values
)


# ------------------------------------------------------------
# Check missing weather mappings
# ------------------------------------------------------------

weather_missing = (
    road_features["grid_id"]
    .isna()
    .sum()
)


print(
    f"\nRoads without weather cell: "
    f"{weather_missing}"
)


if weather_missing > 0:

    raise RuntimeError(
        "Some roads were not mapped to weather cells."
    )


# ------------------------------------------------------------
# Check road count
# ------------------------------------------------------------

print(
    f"\nFinal road rows: "
    f"{len(road_features)}"
)


# ============================================================
# STEP 10
# SAVE ROAD FEATURES
# ============================================================

print("\n==========================================")
print("STEP 9: Saving road features")
print("==========================================")


road_features.to_csv(
    OUTPUT_ROAD_FILE,
    index=False
)


print(
    f"Saved: {OUTPUT_ROAD_FILE}"
)


print(
    f"Rows: {len(road_features)}"
)


# ============================================================
# SHOW SAMPLE
# ============================================================

print("\nFirst 5 rows:")


print(
    road_features[
        [

            "edge_id",

            "grid_id",

            "current_rain_mm",

            "rain_event_day",

            "rain_3d",

            "rain_7d",

            "rain_10d",

            "max_rain_3d",

            "max_rain_7d",

            "heavy_rain_days",

            "elevation_m",

            "slope_deg",

            "distance_to_river_m"

        ]
    ].head()
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("STEP 26 COMPLETE")
print("==========================================")

print(
    "\nOutput files:"
)

print(
    f"1. {OUTPUT_WEATHER_FILE}"
)

print(
    f"2. {OUTPUT_ROAD_FILE}"
)

print(
    "\nReady for Step 27:"
)

print(
    "Live LightGBM prediction + dynamic road risk"
)