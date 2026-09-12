
import os
import time

import numpy as np
import pandas as pd
import psycopg2
import requests


# ============================================================
# 1. Configuration
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "assam_routing",
    "user": "postgres",
    "password": "postgres"
}


OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\live_weather_grid.csv"
)


# ------------------------------------------------------------
# Open-Meteo
# ------------------------------------------------------------

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ------------------------------------------------------------
# Weather grid
#
# 0.10 degree is roughly 10 km north-south.
# This is suitable for the first prototype.
# ------------------------------------------------------------

GRID_STEP = 0.10


# ------------------------------------------------------------
# We need:
#
# current weather
# previous 10 days hourly rainfall
# next 2 days forecast
#
# The previous 10 days allow us to calculate 7-day rainfall
# and rolling rainfall statistics.
# ------------------------------------------------------------

PAST_DAYS = 10

FORECAST_DAYS = 2


# ------------------------------------------------------------
# API batch size
# ------------------------------------------------------------

BATCH_SIZE = 50


# ============================================================
# 2. Print helper
# ============================================================

def print_section(title):

    print(
        "\n"
        + "=" * 60
    )

    print(title)

    print(
        "=" * 60
    )


# ============================================================
# 3. Connect to database
# ============================================================

print_section(
    "READING ROUTING AREA"
)


conn = psycopg2.connect(
    **DB_CONFIG
)

cur = conn.cursor()


# ============================================================
# 4. Get routing road extent
# ============================================================

cur.execute(
    """
    SELECT
        MIN(
            ST_X(
                ST_Transform(
                    ST_StartPoint(geometry),
                    4326
                )
            )
        ) AS min_lon,

        MAX(
            ST_X(
                ST_Transform(
                    ST_StartPoint(geometry),
                    4326
                )
            )
        ) AS max_lon,

        MIN(
            ST_Y(
                ST_Transform(
                    ST_StartPoint(geometry),
                    4326
                )
            )
        ) AS min_lat,

        MAX(
            ST_Y(
                ST_Transform(
                    ST_StartPoint(geometry),
                    4326
                )
            )
        ) AS max_lat

    FROM routing_noded;
    """
)


min_lon, max_lon, min_lat, max_lat = (
    cur.fetchone()
)


cur.close()

conn.close()


# Convert database values to normal Python floats.

min_lon = float(min_lon)

max_lon = float(max_lon)

min_lat = float(min_lat)

max_lat = float(max_lat)


print(
    "Longitude:",
    min_lon,
    "to",
    max_lon
)

print(
    "Latitude:",
    min_lat,
    "to",
    max_lat
)


# ============================================================
# 5. Create weather grid
# ============================================================

print_section(
    "CREATING WEATHER GRID"
)


# Add a tiny margin around the routing area.

grid_min_lon = (
    np.floor(
        min_lon / GRID_STEP
    )
    * GRID_STEP
)

grid_max_lon = (
    np.ceil(
        max_lon / GRID_STEP
    )
    * GRID_STEP
)

grid_min_lat = (
    np.floor(
        min_lat / GRID_STEP
    )
    * GRID_STEP
)

grid_max_lat = (
    np.ceil(
        max_lat / GRID_STEP
    )
    * GRID_STEP
)


longitudes = np.arange(
    grid_min_lon,
    grid_max_lon + GRID_STEP / 2,
    GRID_STEP
)

latitudes = np.arange(
    grid_min_lat,
    grid_max_lat + GRID_STEP / 2,
    GRID_STEP
)


grid_points = []


grid_id = 1


for latitude in latitudes:

    for longitude in longitudes:

        grid_points.append(
            {
                "grid_id": grid_id,
                "longitude": round(
                    float(longitude),
                    4
                ),
                "latitude": round(
                    float(latitude),
                    4
                )
            }
        )

        grid_id += 1


grid_df = pd.DataFrame(
    grid_points
)


print(
    "Grid points:",
    len(grid_df)
)


# ============================================================
# 6. Open-Meteo request helper
# ============================================================

def fetch_weather_batch(
    batch
):

    latitudes_text = ",".join(
        str(
            float(latitude)
        )
        for latitude in batch["latitude"]
    )


    longitudes_text = ",".join(
        str(
            float(longitude)
        )
        for longitude in batch["longitude"]
    )


    params = {

        "latitude":
            latitudes_text,

        "longitude":
            longitudes_text,

        # ----------------------------------------------------
        # Current weather
        # ----------------------------------------------------

        "current":
            ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "rain",
                    "weather_code",
                    "visibility",
                    "wind_speed_10m",
                    "wind_gusts_10m"
                ]
            ),

        # ----------------------------------------------------
        # Hourly history + forecast
        # ----------------------------------------------------

        "hourly":
            ",".join(
                [
                    "precipitation",
                    "rain",
                    "temperature_2m",
                    "visibility",
                    "wind_speed_10m",
                    "wind_gusts_10m"
                ]
            ),

        # ----------------------------------------------------
        # Previous 10 days
        # ----------------------------------------------------

        "past_days":
            PAST_DAYS,

        # ----------------------------------------------------
        # Small forecast window
        # ----------------------------------------------------

        "forecast_days":
            FORECAST_DAYS,

        # ----------------------------------------------------
        # India local time
        # ----------------------------------------------------

        "timezone":
            "Asia/Kolkata",

        # ----------------------------------------------------
        # Standard prototype units
        # ----------------------------------------------------

        "temperature_unit":
            "celsius",

        "wind_speed_unit":
            "kmh",

        "precipitation_unit":
            "mm"
    }


    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )


    response.raise_for_status()

    return response.json()


# ============================================================
# 7. Extract hourly rainfall statistics
# ============================================================

def calculate_features(
    weather
):

    hourly = weather.get(
        "hourly",
        {}
    )


    times = hourly.get(
        "time",
        []
    )


    rain = np.array(
        hourly.get(
            "rain",
            []
        ),
        dtype=float
    )


    precipitation = np.array(
        hourly.get(
            "precipitation",
            []
        ),
        dtype=float
    )


    temperature = np.array(
        hourly.get(
            "temperature_2m",
            []
        ),
        dtype=float
    )


    visibility = np.array(
        hourly.get(
            "visibility",
            []
        ),
        dtype=float
    )


    wind_speed = np.array(
        hourly.get(
            "wind_speed_10m",
            []
        ),
        dtype=float
    )


    wind_gusts = np.array(
        hourly.get(
            "wind_gusts_10m",
            []
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(rain) == 0:

        return {
            "current_rain_mm": np.nan,
            "rain_24h_mm": np.nan,
            "rain_3d_mm": np.nan,
            "rain_7d_mm": np.nan,
            "max_rain_3d_mm": np.nan,
            "max_rain_7d_mm": np.nan,
            "heavy_rain_days": np.nan,
            "current_temperature_c": np.nan,
            "current_visibility_m": np.nan,
            "current_wind_kmh": np.nan,
            "current_wind_gust_kmh": np.nan
        }


    # --------------------------------------------------------
    # Clean hourly arrays
    # --------------------------------------------------------

    rain[
        ~np.isfinite(rain)
    ] = 0.0


    precipitation[
        ~np.isfinite(precipitation)
    ] = 0.0


    # --------------------------------------------------------
    # The API returns past + current + forecast.
    #
    # We want only the observations up to "now" for the
    # historical accumulation features.
    #
    # The current weather object is used separately.
    # --------------------------------------------------------

    current = weather.get(
        "current",
        {}
    )


    current_time = current.get(
        "time"
    )


    # Find current time in hourly data.

    if (
        current_time is not None
        and
        current_time in times
    ):

        current_index = times.index(
            current_time
        )

    else:

        # Fallback:
        # use the last past/current hour.
        current_index = len(rain) - 1


    historical_rain = rain[
        :current_index + 1
    ]


    # --------------------------------------------------------
    # Current hourly rainfall
    # --------------------------------------------------------

    if len(historical_rain) > 0:

        current_rain = float(
            historical_rain[-1]
        )

    else:

        current_rain = 0.0


    # --------------------------------------------------------
    # Recent rainfall totals
    #
    # 24 hourly observations ≈ 24 hours
    # 72 hourly observations ≈ 3 days
    # 168 hourly observations ≈ 7 days
    # --------------------------------------------------------

    rain_24h = float(
        np.sum(
            historical_rain[-24:]
        )
    )


    rain_3d = float(
        np.sum(
            historical_rain[-72:]
        )
    )


    rain_7d = float(
        np.sum(
            historical_rain[-168:]
        )
    )


    # --------------------------------------------------------
    # Rolling 3-day / 7-day maximum rainfall
    #
    # These are rolling totals in mm.
    # --------------------------------------------------------

    max_rain_3d = 0.0

    max_rain_7d = 0.0


    for i in range(
        len(historical_rain)
    ):

        window_3 = historical_rain[
            max(
                0,
                i - 71
            ):
            i + 1
        ]


        window_7 = historical_rain[
            max(
                0,
                i - 167
            ):
            i + 1
        ]


        total_3 = np.sum(
            window_3
        )


        total_7 = np.sum(
            window_7
        )


        max_rain_3d = max(
            max_rain_3d,
            float(total_3)
        )


        max_rain_7d = max(
            max_rain_7d,
            float(total_7)
        )


    # --------------------------------------------------------
    # Daily rainfall for heavy-rain days
    # --------------------------------------------------------

    daily_dataframe = pd.DataFrame(
        {
            "time": pd.to_datetime(
                times[:current_index + 1]
            ),
            "rain": historical_rain
        }
    )


    if len(
        daily_dataframe
    ) > 0:

        daily_rain = (
            daily_dataframe
            .set_index("time")
            ["rain"]
            .resample("D")
            .sum()
        )


        heavy_rain_days = int(
            (
                daily_rain
                >= 50.0
            ).sum()
        )

    else:

        heavy_rain_days = 0


    # --------------------------------------------------------
    # Current non-rain variables
    # --------------------------------------------------------

    current_temperature = current.get(
        "temperature_2m",
        np.nan
    )


    current_visibility = current.get(
        "visibility",
        np.nan
    )


    current_wind = current.get(
        "wind_speed_10m",
        np.nan
    )


    current_gust = current.get(
        "wind_gusts_10m",
        np.nan
    )


    return {
        "current_rain_mm":
            float(
                current_rain
            ),

        "rain_24h_mm":
            rain_24h,

        "rain_3d_mm":
            rain_3d,

        "rain_7d_mm":
            rain_7d,

        "max_rain_3d_mm":
            max_rain_3d,

        "max_rain_7d_mm":
            max_rain_7d,

        "heavy_rain_days":
            heavy_rain_days,

        "current_temperature_c":
            float(
                current_temperature
            ),

        "current_visibility_m":
            float(
                current_visibility
            ),

        "current_wind_kmh":
            float(
                current_wind
            ),

        "current_wind_gust_kmh":
            float(
                current_gust
            )
    }


# ============================================================
# 8. Fetch weather
# ============================================================

print_section(
    "FETCHING LIVE WEATHER"
)


weather_rows = []


for start in range(
    0,
    len(grid_df),
    BATCH_SIZE
):

    batch = grid_df.iloc[
        start:
        start + BATCH_SIZE
    ]


    try:

        response = fetch_weather_batch(
            batch
        )


        # Multiple coordinates produce a list response.

        if isinstance(
            response,
            list
        ):

            locations = response

        else:

            locations = [
                response
            ]


        for index, weather in enumerate(
            locations
        ):

            if index >= len(batch):

                break


            grid_row = batch.iloc[
                index
            ]


            features = calculate_features(
                weather
            )


            weather_rows.append(
                {
                    "grid_id":
                        int(
                            grid_row["grid_id"]
                        ),

                    "longitude":
                        float(
                            grid_row["longitude"]
                        ),

                    "latitude":
                        float(
                            grid_row["latitude"]
                        ),

                    **features
                }
            )


    except Exception as e:

        print(
            "\nWeather request failed for batch",
            start,
            "to",
            start + len(batch) - 1
        )

        print(
            "Error:",
            e
        )


    print(
        "Processed weather:",
        min(
            start + BATCH_SIZE,
            len(grid_df)
        ),
        "/",
        len(grid_df)
    )


    # Small delay to avoid sending requests too rapidly.

    time.sleep(
        0.2
    )


# ============================================================
# 9. Create dataframe
# ============================================================

weather_df = pd.DataFrame(
    weather_rows
)


# ============================================================
# 10. Validate
# ============================================================

print_section(
    "WEATHER VALIDATION"
)


print(
    "Weather grid points:",
    len(weather_df)
)


print(
    "\nMissing values:"
)

print(
    weather_df.isna().sum()
)


# ============================================================
# 11. Save
# ============================================================

weather_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 12. Summary
# ============================================================

print_section(
    "LIVE WEATHER FETCH COMPLETED"
)


print(
    "Grid points saved:",
    len(weather_df)
)

print(
    "Output:",
    OUTPUT_CSV
)


if len(weather_df) > 0:

    print(
        "\nCurrent rainfall summary:"
    )

    print(
        weather_df[
            [
                "current_rain_mm",
                "rain_24h_mm",
                "rain_3d_mm",
                "rain_7d_mm",
                "max_rain_3d_mm",
                "max_rain_7d_mm"
            ]
        ].describe()
    )

    print(
        "\nFirst 10 weather points:"
    )

    print(
        weather_df.head(
            10
        ).to_string(
            index=False
        )
    )

