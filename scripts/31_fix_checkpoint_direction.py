import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\NERProject"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "route_checkpoints.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "route_checkpoints_final.csv"
)


# ============================================================
# READ CHECKPOINTS
# ============================================================

print("\n==========================================")
print("STEP 31: Fix checkpoint direction")
print("==========================================")


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        INPUT_FILE
    )


df = pd.read_csv(
    INPUT_FILE
)


if df.empty:

    raise RuntimeError(
        "Checkpoint file is empty."
    )


print(
    f"Original checkpoints: {len(df)}"
)


# ============================================================
# REVERSE ORDER
# ============================================================

print(
    "\nReversing checkpoint direction..."
)


df = df.sort_values(
    "route_order",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# REBUILD ROUTE ORDER
# ============================================================

df["route_order"] = range(
    1,
    len(df) + 1
)


# ============================================================
# REBUILD DISTANCE FROM START
# ============================================================

original_total_distance = float(
    df["distance_from_start_km"].max()
)


df["distance_from_start_km"] = (

    original_total_distance

    -

    df["distance_from_start_km"]

)


df["distance_from_start_km"] = (

    df[
        "distance_from_start_km"
    ]

    .round(3)

)


# ============================================================
# REBUILD CHECKPOINT IDS
# ============================================================

df["checkpoint_id"] = [

    f"CP-{i:03d}"

    for i in range(
        1,
        len(df) + 1
    )

]


# ============================================================
# SORT AGAIN
# ============================================================

df = df.sort_values(
    "route_order"
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SHOW RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "FINAL CHECKPOINTS"
)

print(
    "=========================================="
)


print(
    df[
        [
            "checkpoint_id",
            "route_order",
            "distance_from_start_km",
            "longitude",
            "latitude",
            "p_landslide",
            "p_flood",
            "p_hazard",
            "hazard_level"
        ]
    ].head(
        10
    ).to_string(
        index=False
    )
)


print(
    "\nLast checkpoints:"
)


print(
    df[
        [
            "checkpoint_id",
            "route_order",
            "distance_from_start_km",
            "longitude",
            "latitude",
            "p_landslide",
            "p_flood",
            "p_hazard",
            "hazard_level"
        ]
    ].tail(
        5
    ).to_string(
        index=False
    )
)


# ============================================================
# VERIFY START / END
# ============================================================

first = df.iloc[0]

last = df.iloc[-1]


print(
    "\n=========================================="
)

print(
    "DIRECTION CHECK"
)

print(
    "=========================================="
)


print(
    f"First checkpoint: "
    f"{first['checkpoint_id']}"
)


print(
    f"First coordinate: "
    f"{first['longitude']:.6f}, "
    f"{first['latitude']:.6f}"
)


print(
    f"Last checkpoint: "
    f"{last['checkpoint_id']}"
)


print(
    f"Last coordinate: "
    f"{last['longitude']:.6f}, "
    f"{last['latitude']:.6f}"
)


print(
    f"Total route distance: "
    f"{df['distance_from_start_km'].max():.3f} km"
)


# ============================================================
# CHECK GAPS
# ============================================================

if len(df) > 1:

    gaps = (

        df[
            "distance_from_start_km"
        ]

        .diff()

        .dropna()

    )

    print(
        f"\nMaximum checkpoint gap: "
        f"{gaps.max():.3f} km"
    )


    if gaps.max() <= 7.0:

        print(
            "All checkpoint gaps remain within 7 km."
        )

    else:

        print(
            "WARNING: checkpoint gap exceeds 7 km."
        )


# ============================================================
# FINAL
# ============================================================

print(
    "\n=========================================="
)

print(
    "STEP 31 COMPLETE"
)

print(
    "=========================================="
)


print(
    "\nFinal checkpoint file:"
)


print(
    OUTPUT_FILE
)