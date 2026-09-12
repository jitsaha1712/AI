
import psycopg2
import pandas as pd


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "assam_routing",
    "user": "postgres",
    "password": "postgres"   # use your existing PostgreSQL password
}


# ============================================================
# SOURCE / DESTINATION
# ============================================================

SOURCE_LON = 91.7362
SOURCE_LAT = 26.1445

DEST_LON = 92.6836
DEST_LAT = 26.3509


# ============================================================
# CHECKPOINT SETTINGS
# ============================================================

CHECKPOINT_DISTANCE_KM = 6.0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = (
    r"C:\NERProject\Data\processed"
    r"\route_checkpoints.csv"
)


# ============================================================
# CONNECT
# ============================================================

print("\n==========================================")
print("STEP 30: Create route checkpoints")
print("==========================================")


conn = psycopg2.connect(
    **DB_CONFIG
)


try:

    # ========================================================
    # STEP 30A
    # FIND SOURCE NODE
    # ========================================================

    print(
        "\nFinding source node..."
    )


    source_sql = """
    SELECT
        node_id
    FROM routing_vertices
    ORDER BY
        geometry <-> ST_Transform(
            ST_SetSRID(
                ST_MakePoint(%s, %s),
                4326
            ),
            32645
        )
    LIMIT 1;
    """


    with conn.cursor() as cur:

        cur.execute(
            source_sql,
            (
                SOURCE_LON,
                SOURCE_LAT
            )
        )

        source_result = cur.fetchone()


    if source_result is None:

        raise RuntimeError(
            "Source node not found."
        )


    source_node = int(
        source_result[0]
    )


    print(
        f"Source node: {source_node}"
    )


    # ========================================================
    # STEP 30B
    # FIND DESTINATION NODE
    # ========================================================

    print(
        "\nFinding destination node..."
    )


    destination_sql = """
    SELECT
        node_id
    FROM routing_vertices
    ORDER BY
        geometry <-> ST_Transform(
            ST_SetSRID(
                ST_MakePoint(%s, %s),
                4326
            ),
            32645
        )
    LIMIT 1;
    """


    with conn.cursor() as cur:

        cur.execute(
            destination_sql,
            (
                DEST_LON,
                DEST_LAT
            )
        )

        destination_result = cur.fetchone()


    if destination_result is None:

        raise RuntimeError(
            "Destination node not found."
        )


    destination_node = int(
        destination_result[0]
    )


    print(
        f"Destination node: {destination_node}"
    )


    # ========================================================
    # STEP 30C
    # CALCULATE CURRENT DYNAMIC ROUTE
    # ========================================================

    print(
        "\nCalculating dynamic risk-aware route..."
    )


    route_sql = """

    SELECT *

    FROM pgr_dijkstra(

        $$

        SELECT

            r.edge_id AS id,

            r.source AS source,

            r.target AS target,

            d.dynamic_weight AS cost,

            d.dynamic_weight AS reverse_cost

        FROM routing_noded AS r

        INNER JOIN dynamic_edge_state AS d

            ON d.edge_id = r.edge_id

        WHERE d.dynamic_weight IS NOT NULL

          AND d.dynamic_weight > 0

        $$,

        %s,

        %s,

        directed := TRUE

    );

    """


    with conn.cursor() as cur:

        cur.execute(
            route_sql,
            (
                source_node,
                destination_node
            )
        )

        route_rows = cur.fetchall()

        route_columns = [

            desc[0]

            for desc in cur.description

        ]


    if len(route_rows) == 0:

        raise RuntimeError(
            "No dynamic route found."
        )


    route_df = pd.DataFrame(
        route_rows,
        columns=route_columns
    )


    # ========================================================
    # GET ROUTE EDGE IDS
    # ========================================================

    route_edge_ids = [

        int(edge)

        for edge in route_df["edge"]

        if int(edge) != -1

    ]


    if len(route_edge_ids) == 0:

        raise RuntimeError(
            "Dynamic route contains no edges."
        )


    print(
        f"Route edges: {len(route_edge_ids)}"
    )


    # ========================================================
    # STEP 30D
    # BUILD ORDERED ROUTE GEOMETRY
    # ========================================================

    print(
        "\nBuilding route geometry..."
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # routing_noded uses column name:
    #
    # geometry
    #
    # We explicitly create the alias route_geom.
    # --------------------------------------------------------

    geometry_sql = """

    WITH ordered_edges AS (

        SELECT

            edge_id,

            geometry,

            array_position(
                %s::bigint[],
                edge_id
            ) AS route_order

        FROM routing_noded

        WHERE edge_id = ANY(%s)

    ),

    route_geometry AS (

        SELECT

            ST_LineMerge(

                ST_Union(
                    geometry
                )

            ) AS route_geom

        FROM ordered_edges

    )

    SELECT

        route_geom,

        ST_Length(
            route_geom
        ) / 1000.0 AS route_length_km

    FROM route_geometry;

    """


    with conn.cursor() as cur:

        cur.execute(

            geometry_sql,

            (
                route_edge_ids,
                route_edge_ids
            )

        )

        geometry_result = cur.fetchone()


    if geometry_result is None:

        raise RuntimeError(
            "Could not create route geometry."
        )


    route_geom = geometry_result[0]

    route_length_km = float(
        geometry_result[1]
    )


    if route_geom is None:

        raise RuntimeError(
            "Route geometry is NULL."
        )


    print(
        f"Route geometry length: "
        f"{route_length_km:.3f} km"
    )


    # ========================================================
    # STEP 30E
    # GENERATE CHECKPOINT DISTANCES
    # ========================================================

    print(
        "\nGenerating checkpoints every "
        f"{CHECKPOINT_DISTANCE_KM:.1f} km..."
    )


    checkpoint_distances = []


    distance = 0.0


    while distance < route_length_km:

        checkpoint_distances.append(
            distance
        )

        distance += CHECKPOINT_DISTANCE_KM


    # --------------------------------------------------------
    # Always include final destination.
    # --------------------------------------------------------

    if (

        len(checkpoint_distances) == 0

        or

        abs(
            checkpoint_distances[-1]
            -
            route_length_km
        ) > 0.000001

    ):

        checkpoint_distances.append(
            route_length_km
        )


    # ========================================================
    # STEP 30F
    # CREATE CHECKPOINTS
    # ========================================================

    checkpoint_rows = []


    for index, distance_km in enumerate(
        checkpoint_distances
    ):


        # ----------------------------------------------------
        # Convert distance to fraction of route.
        # ----------------------------------------------------

        if route_length_km > 0:

            fraction = (

                distance_km
                /
                route_length_km

            )

        else:

            fraction = 0.0


        # Prevent tiny floating-point issues.

        fraction = max(
            0.0,
            min(
                1.0,
                fraction
            )
        )


        # ====================================================
        # GET POINT ON ROUTE
        # ====================================================

        point_sql = """

        SELECT

            ST_X(

                ST_Transform(

                    ST_LineInterpolatePoint(
                        %s,
                        %s
                    ),

                    4326

                )

            ) AS longitude,


            ST_Y(

                ST_Transform(

                    ST_LineInterpolatePoint(
                        %s,
                        %s
                    ),

                    4326

                )

            ) AS latitude;

        """


        with conn.cursor() as cur:

            cur.execute(

                point_sql,

                (
                    route_geom,
                    fraction,
                    route_geom,
                    fraction
                )

            )

            point_result = cur.fetchone()


        if point_result is None:

            continue


        longitude = float(
            point_result[0]
        )


        latitude = float(
            point_result[1]
        )


        # ====================================================
        # FIND NEAREST ROUTE EDGE
        # ====================================================

        risk_sql = """

        SELECT

            d.p_landslide,

            d.p_flood,

            d.p_hazard,

            d.dynamic_weight

        FROM routing_noded AS r

        INNER JOIN dynamic_edge_state AS d

            ON d.edge_id = r.edge_id

        WHERE r.edge_id = ANY(%s)

        ORDER BY

            ST_Distance(

                r.geometry,

                ST_Transform(

                    ST_SetSRID(

                        ST_MakePoint(
                            %s,
                            %s
                        ),

                        4326

                    ),

                    32645

                )

            )

        LIMIT 1;

        """


        with conn.cursor() as cur:

            cur.execute(

                risk_sql,

                (
                    route_edge_ids,
                    longitude,
                    latitude
                )

            )

            risk_result = cur.fetchone()


        if risk_result is None:

            p_landslide = None

            p_flood = None

            p_hazard = None

            dynamic_weight = None

        else:

            p_landslide = float(
                risk_result[0]
            )

            p_flood = float(
                risk_result[1]
            )

            p_hazard = float(
                risk_result[2]
            )

            dynamic_weight = float(
                risk_result[3]
            )


        # ====================================================
        # HAZARD LEVEL
        # ====================================================

        if p_hazard is None:

            hazard_level = "unknown"

        elif p_hazard < 0.20:

            hazard_level = "low"

        elif p_hazard < 0.40:

            hazard_level = "medium"

        elif p_hazard < 0.60:

            hazard_level = "high"

        else:

            hazard_level = "very_high"


        # ====================================================
        # STORE CHECKPOINT
        # ====================================================

        checkpoint_rows.append({

            "checkpoint_id":
                f"CP-{index + 1:03d}",

            "route_order":
                index + 1,

            "distance_from_start_km":
                round(
                    distance_km,
                    3
                ),

            "longitude":
                longitude,

            "latitude":
                latitude,

            "p_landslide":
                p_landslide,

            "p_flood":
                p_flood,

            "p_hazard":
                p_hazard,

            "dynamic_weight":
                dynamic_weight,

            "hazard_level":
                hazard_level

        })


    # ========================================================
    # STEP 30G
    # CREATE DATAFRAME
    # ========================================================

    checkpoints = pd.DataFrame(
        checkpoint_rows
    )


    if checkpoints.empty:

        raise RuntimeError(
            "No checkpoints were generated."
        )


    # ========================================================
    # STEP 30H
    # SAVE CSV
    # ========================================================

    checkpoints.to_csv(

        OUTPUT_FILE,

        index=False

    )


    # ========================================================
    # STEP 30I
    # DISPLAY SUMMARY
    # ========================================================

    print(
        "\n=========================================="
    )


    print(
        "CHECKPOINT SUMMARY"
    )


    print(
        "=========================================="
    )


    print(
        f"Route length     : "
        f"{route_length_km:.3f} km"
    )


    print(
        f"Checkpoint count : "
        f"{len(checkpoints)}"
    )


    print(
        f"Target spacing   : "
        f"{CHECKPOINT_DISTANCE_KM:.1f} km"
    )


    print(
        "\nFirst checkpoints:"
    )


    print(

        checkpoints.head(
            10
        ).to_string(
            index=False
        )

    )


    print(
        "\nLast checkpoints:"
    )


    print(

        checkpoints.tail(
            5
        ).to_string(
            index=False
        )

    )


    # ========================================================
    # STEP 30J
    # CHECK CHECKPOINT GAPS
    # ========================================================

    if len(checkpoints) > 1:

        gaps = (

            checkpoints[
                "distance_from_start_km"
            ]

            .diff()

            .dropna()

        )


        print(
            "\nCheckpoint gap statistics:"
        )


        print(
            gaps.describe()
        )


        maximum_gap = float(
            gaps.max()
        )


        print(
            f"\nMaximum checkpoint gap: "
            f"{maximum_gap:.3f} km"
        )


        if maximum_gap <= 7.0:

            print(
                "All checkpoint gaps are within 7 km."
            )

        else:

            print(
                "WARNING: checkpoint gap exceeds 7 km."
            )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n=========================================="
    )


    print(
        "STEP 30 COMPLETE"
    )


    print(
        "=========================================="
    )


    print(
        "\nCheckpoint file:"
    )


    print(
        OUTPUT_FILE
    )


finally:

    conn.close()

