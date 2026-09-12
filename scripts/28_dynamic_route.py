
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
# TEST SOURCE / DESTINATION
# ============================================================

SOURCE_LON = 91.7362
SOURCE_LAT = 26.1445

DEST_LON = 92.6836
DEST_LAT = 26.3509


# ============================================================
# CONNECT
# ============================================================

print("\n==========================================")
print("STEP 28: Dynamic pgRouting")
print("==========================================")


try:

    conn = psycopg2.connect(
        **DB_CONFIG
    )

except Exception as e:

    raise RuntimeError(
        f"Could not connect to PostgreSQL:\n{e}"
    )


try:

    # ========================================================
    # STEP 28A
    # INSPECT routing_vertices
    # ========================================================

    print(
        "\nInspecting routing_vertices schema..."
    )


    vertex_schema_sql = """
    SELECT
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'routing_vertices'
    ORDER BY ordinal_position;
    """


    with conn.cursor() as cur:

        cur.execute(
            vertex_schema_sql
        )

        vertex_schema = cur.fetchall()


    if len(vertex_schema) == 0:

        raise RuntimeError(
            "routing_vertices table was not found."
        )


    vertex_columns = [

        row[0]

        for row in vertex_schema

    ]


    print(
        "\nrouting_vertices columns:"
    )


    for column_name, data_type in vertex_schema:

        print(
            f"  {column_name} : {data_type}"
        )


    # ========================================================
    # FIND VERTEX ID
    # ========================================================

    possible_vertex_id_columns = [

        "vertex_id",
        "node_id",
        "vid",
        "id"

    ]


    vertex_id_column = None


    for candidate in possible_vertex_id_columns:

        if candidate in vertex_columns:

            vertex_id_column = candidate

            break


    if vertex_id_column is None:

        raise RuntimeError(

            "Could not identify vertex ID column.\n"
            f"Available columns: {vertex_columns}"

        )


    print(
        f"\nUsing vertex ID column: "
        f"{vertex_id_column}"
    )


    # ========================================================
    # FIND VERTEX GEOMETRY
    # ========================================================

    possible_vertex_geometry_columns = [

        "geom",
        "geometry",
        "the_geom"

    ]


    vertex_geometry_column = None


    for candidate in possible_vertex_geometry_columns:

        if candidate in vertex_columns:

            vertex_geometry_column = candidate

            break


    if vertex_geometry_column is None:

        raise RuntimeError(

            "Could not identify geometry column in "
            "routing_vertices.\n"
            f"Available columns: {vertex_columns}"

        )


    print(
        f"Using geometry column: "
        f"{vertex_geometry_column}"
    )


    # ========================================================
    # STEP 28B
    # INSPECT routing_noded
    # ========================================================

    print(
        "\nInspecting routing_noded schema..."
    )


    edge_schema_sql = """
    SELECT
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'routing_noded'
    ORDER BY ordinal_position;
    """


    with conn.cursor() as cur:

        cur.execute(
            edge_schema_sql
        )

        edge_schema = cur.fetchall()


    if len(edge_schema) == 0:

        raise RuntimeError(
            "routing_noded table was not found."
        )


    edge_columns = [

        row[0]

        for row in edge_schema

    ]


    print(
        "\nrouting_noded columns:"
    )


    for column_name, data_type in edge_schema:

        print(
            f"  {column_name} : {data_type}"
        )


    # ========================================================
    # FIND EDGE ID
    # ========================================================

    if "edge_id" not in edge_columns:

        raise RuntimeError(
            "routing_noded does not contain edge_id."
        )


    edge_id_column = "edge_id"


    # ========================================================
    # FIND SOURCE / TARGET
    # ========================================================

    possible_source_columns = [

        "source",
        "source_node",
        "source_vid",
        "from_node",
        "from_vertex",
        "start_node"

    ]


    possible_target_columns = [

        "target",
        "target_node",
        "target_vid",
        "to_node",
        "to_vertex",
        "end_node"

    ]


    source_column = None

    target_column = None


    for candidate in possible_source_columns:

        if candidate in edge_columns:

            source_column = candidate

            break


    for candidate in possible_target_columns:

        if candidate in edge_columns:

            target_column = candidate

            break


    if source_column is None:

        raise RuntimeError(

            "Could not identify source column in routing_noded.\n"
            f"Available columns: {edge_columns}"

        )


    if target_column is None:

        raise RuntimeError(

            "Could not identify target column in routing_noded.\n"
            f"Available columns: {edge_columns}"

        )


    print(
        f"\nUsing edge ID column: {edge_id_column}"
    )


    print(
        f"Using source column: {source_column}"
    )


    print(
        f"Using target column: {target_column}"
    )


    # ========================================================
    # STEP 28C
    # CHECK dynamic_edge_state
    # ========================================================

    print(
        "\nChecking dynamic_edge_state..."
    )


    dynamic_schema_sql = """
    SELECT
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'dynamic_edge_state'
    ORDER BY ordinal_position;
    """


    with conn.cursor() as cur:

        cur.execute(
            dynamic_schema_sql
        )

        dynamic_schema = cur.fetchall()


    if len(dynamic_schema) == 0:

        raise RuntimeError(
            "dynamic_edge_state table was not found."
        )


    dynamic_columns = [

        row[0]

        for row in dynamic_schema

    ]


    required_dynamic_columns = [

        "edge_id",
        "dynamic_weight",
        "p_landslide",
        "p_flood",
        "p_hazard"

    ]


    missing_dynamic = [

        col

        for col in required_dynamic_columns

        if col not in dynamic_columns

    ]


    if len(missing_dynamic) > 0:

        raise RuntimeError(

            "dynamic_edge_state is missing:\n"

            +
            "\n".join(
                missing_dynamic
            )

        )


    print(
        "Required dynamic risk columns are present."
    )


    # ========================================================
    # STEP 28D
    # VERIFY DYNAMIC STATE COVERAGE
    # ========================================================

    print(
        "\nChecking dynamic state coverage..."
    )


    coverage_sql = """

    SELECT

        COUNT(*) AS routing_edges,

        COUNT(d.edge_id) AS dynamic_edges,

        COUNT(
            CASE
                WHEN d.dynamic_weight IS NOT NULL
                THEN 1
            END
        ) AS weighted_dynamic_edges

    FROM routing_noded AS r

    LEFT JOIN dynamic_edge_state AS d

        ON d.edge_id = r.edge_id;

    """


    with conn.cursor() as cur:

        cur.execute(
            coverage_sql
        )

        coverage = cur.fetchone()


    routing_edge_count = int(
        coverage[0]
    )


    dynamic_edge_count = int(
        coverage[1]
    )


    weighted_dynamic_count = int(
        coverage[2]
    )


    print(
        f"Routing edges       : "
        f"{routing_edge_count}"
    )


    print(
        f"Dynamic state edges : "
        f"{dynamic_edge_count}"
    )


    print(
        f"Dynamic weighted    : "
        f"{weighted_dynamic_count}"
    )


    if dynamic_edge_count != routing_edge_count:

        raise RuntimeError(

            "dynamic_edge_state does not cover "
            "every routing edge."

        )


    if weighted_dynamic_count != routing_edge_count:

        raise RuntimeError(

            "Some routing edges do not have "
            "dynamic_weight."

        )


    # ========================================================
    # STEP 28E
    # FIND SOURCE NODE
    # ========================================================

    print(
        "\nFinding nearest source node..."
    )


    source_sql = f"""

    SELECT

        "{vertex_id_column}" AS vertex_id

    FROM routing_vertices

    ORDER BY

        "{vertex_geometry_column}" <->

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
            "Source node could not be found."
        )


    source_node = int(
        source_result[0]
    )


    print(
        f"Source node: {source_node}"
    )


    # ========================================================
    # STEP 28F
    # FIND DESTINATION NODE
    # ========================================================

    print(
        "\nFinding nearest destination node..."
    )


    destination_sql = f"""

    SELECT

        "{vertex_id_column}" AS vertex_id

    FROM routing_vertices

    ORDER BY

        "{vertex_geometry_column}" <->

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
            "Destination node could not be found."
        )


    destination_node = int(
        destination_result[0]
    )


    print(
        f"Destination node: {destination_node}"
    )


    # ========================================================
    # STEP 28G
    # CHECK SOURCE / DESTINATION
    # ========================================================

    print(
        "\nChecking source and destination..."
    )


    node_check_sql = f"""

    SELECT COUNT(*)

    FROM routing_vertices

    WHERE "{vertex_id_column}" IN (%s, %s);

    """


    with conn.cursor() as cur:

        cur.execute(

            node_check_sql,

            (
                source_node,
                destination_node
            )

        )

        node_count = cur.fetchone()[0]


    if node_count != 2:

        raise RuntimeError(
            "Source or destination node is invalid."
        )


    print(
        "Source and destination nodes are valid."
    )


    # ========================================================
    # STEP 28H
    # RUN DYNAMIC pgr_dijkstra
    # ========================================================

    print(
        "\nRunning dynamic pgRouting..."
    )


    # IMPORTANT:
    #
    # We now use dynamic_edge_state.dynamic_weight.
    #
    # Before this fix, Step 28 incorrectly used the old
    # routing_noded.dynamic_weight generated during Step 24.
    #
    # Now:
    #
    # routing_noded
    #      +
    # dynamic_edge_state
    #      ↓
    # current dynamic weight
    #


    route_sql = f"""

    SELECT *

    FROM pgr_dijkstra(

        $$

        SELECT

            r.edge_id AS id,

            r."{source_column}" AS source,

            r."{target_column}" AS target,

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

            "No dynamic route found between "
            f"{source_node} and {destination_node}."

        )


    route_df = pd.DataFrame(

        route_rows,

        columns=route_columns

    )


    print(
        "\nDynamic route found!"
    )


    print(
        f"Route result rows: "
        f"{len(route_df)}"
    )


    print(
        "\npgr_dijkstra columns:"
    )


    print(
        list(
            route_df.columns
        )
    )


    # ========================================================
    # STEP 28I
    # EXTRACT ROUTE EDGE IDS
    # ========================================================

    route_edge_ids = []


    for edge_id in route_df["edge"]:

        edge_id = int(
            edge_id
        )


        if edge_id != -1:

            route_edge_ids.append(
                edge_id
            )


    if len(route_edge_ids) == 0:

        raise RuntimeError(
            "Route contains no valid edges."
        )


    print(
        f"Valid route edges: "
        f"{len(route_edge_ids)}"
    )


    # ========================================================
    # STEP 28J
    # GET CURRENT ROUTE RISK
    # ========================================================

    print(
        "\nReading current route risk..."
    )


    route_edges_sql = """

    SELECT

        r.edge_id,

        r.source,

        r.target,

        r.length_km,

        d.dynamic_weight,

        d.p_landslide,

        d.p_flood,

        d.p_hazard

    FROM routing_noded AS r

    INNER JOIN dynamic_edge_state AS d

        ON d.edge_id = r.edge_id

    WHERE r.edge_id = ANY(%s);

    """


    with conn.cursor() as cur:

        cur.execute(

            route_edges_sql,

            (
                route_edge_ids,
            )

        )

        edge_rows = cur.fetchall()

        edge_columns = [

            desc[0]

            for desc in cur.description

        ]


    edge_df = pd.DataFrame(

        edge_rows,

        columns=edge_columns

    )


    if edge_df.empty:

        raise RuntimeError(
            "Could not read route edge information."
        )


    # ========================================================
    # PRESERVE ROUTE ORDER
    # ========================================================

    route_order = {

        edge_id: position

        for position, edge_id

        in enumerate(route_edge_ids)

    }


    edge_df[
        "route_order"
    ] = (

        edge_df[
            "edge_id"
        ]

        .map(
            route_order
        )

    )


    edge_df = (

        edge_df

        .sort_values(
            "route_order"
        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # STEP 28K
    # ROUTE SUMMARY
    # ========================================================

    total_distance = float(

        edge_df[
            "length_km"
        ].sum()

    )


    total_dynamic_cost = float(

        edge_df[
            "dynamic_weight"
        ].sum()

    )


    average_landslide = float(

        edge_df[
            "p_landslide"
        ].mean()

    )


    average_flood = float(

        edge_df[
            "p_flood"
        ].mean()

    )


    average_hazard = float(

        edge_df[
            "p_hazard"
        ].mean()

    )


    maximum_hazard = float(

        edge_df[
            "p_hazard"
        ].max()

    )


    landslide_alert_edges = int(

        (

            edge_df[
                "p_landslide"
            ]

            >= 0.20

        ).sum()

    )


    # ========================================================
    # FINAL ROUTE SUMMARY
    # ========================================================

    print(
        "\n=========================================="
    )


    print(
        "DYNAMIC ROUTE SUMMARY"
    )


    print(
        "=========================================="
    )


    print(
        f"Source node          : "
        f"{source_node}"
    )


    print(
        f"Destination node     : "
        f"{destination_node}"
    )


    print(
        f"Route edges          : "
        f"{len(edge_df)}"
    )


    print(
        f"Physical distance    : "
        f"{total_distance:.3f} km"
    )


    print(
        f"Dynamic cost         : "
        f"{total_dynamic_cost:.3f}"
    )


    print(
        f"Average landslide P  : "
        f"{average_landslide:.4f}"
    )


    print(
        f"Average flood P      : "
        f"{average_flood:.4f}"
    )


    print(
        f"Average hazard P     : "
        f"{average_hazard:.4f}"
    )


    print(
        f"Maximum hazard P     : "
        f"{maximum_hazard:.4f}"
    )


    print(
        f"Landslide alerts     : "
        f"{landslide_alert_edges}"
    )


    # ========================================================
    # HAZARD SUMMARY
    # ========================================================

    print(
        "\nHazard distribution on route:"
    )


    hazard_bins = pd.cut(

        edge_df[
            "p_hazard"
        ],

        bins=[

            0.0,
            0.20,
            0.40,
            0.60,
            0.80,
            1.0

        ],

        include_lowest=True

    )


    print(
        hazard_bins
        .value_counts()
        .sort_index()
        .to_string()
    )


    # ========================================================
    # TOP 10 RISKIEST SEGMENTS
    # ========================================================

    print(
        "\nTop 10 riskiest route segments:"
    )


    top_risky = (

        edge_df[
            [

                "edge_id",

                "length_km",

                "p_landslide",

                "p_flood",

                "p_hazard",

                "dynamic_weight"

            ]

        ]

        .sort_values(

            "p_hazard",

            ascending=False

        )

        .head(10)

    )


    print(
        top_risky.to_string(
            index=False
        )
    )


finally:

    conn.close()


# ============================================================
# FINAL
# ============================================================

print(
    "\n=========================================="
)


print(
    "STEP 28 COMPLETE"
)


print(
    "=========================================="
)


print(
    "\nDynamic pgRouting used "
    "dynamic_edge_state.dynamic_weight."
)


print(
    "\nNext step:"
)


print(
    "Step 29: Compare shortest route vs "
    "dynamic safest route."
)

