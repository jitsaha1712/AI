
import psycopg2
import pandas as pd


# ============================================================
# DATABASE
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
# CONNECT
# ============================================================

print("\n==========================================")
print("STEP 29: Shortest vs Dynamic Route")
print("==========================================")


conn = psycopg2.connect(
    **DB_CONFIG
)


try:

    # ========================================================
    # STEP 29A
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
    # STEP 29B
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
    # HELPER FUNCTION
    # GET ROUTE USING A GIVEN COST
    # ========================================================

    def run_route(cost_expression):


        sql = f"""

        SELECT *

        FROM pgr_dijkstra(

            $$

            SELECT

                r.edge_id AS id,

                r.source AS source,

                r.target AS target,

                {cost_expression} AS cost,

                {cost_expression} AS reverse_cost

            FROM routing_noded AS r

            INNER JOIN dynamic_edge_state AS d

                ON d.edge_id = r.edge_id

            WHERE {cost_expression} IS NOT NULL

              AND {cost_expression} > 0

            $$,

            %s,

            %s,

            directed := TRUE

        );

        """


        with conn.cursor() as cur:

            cur.execute(

                sql,

                (
                    source_node,
                    destination_node
                )

            )

            rows = cur.fetchall()

            columns = [

                desc[0]

                for desc in cur.description

            ]


        if len(rows) == 0:

            raise RuntimeError(
                "No route found."
            )


        return pd.DataFrame(
            rows,
            columns=columns
        )


    # ========================================================
    # HELPER FUNCTION
    # CALCULATE ROUTE STATISTICS
    # ========================================================

    def get_route_statistics(
        route_df,
        route_name
    ):

        # ----------------------------------------------------
        # Remove pgRouting terminal edge (-1)
        # ----------------------------------------------------

        route_edges = [

            int(edge)

            for edge in route_df["edge"]

            if int(edge) != -1

        ]


        if len(route_edges) == 0:

            raise RuntimeError(
                f"{route_name}: route contains no edges."
            )


        # ----------------------------------------------------
        # Read route edge information
        # ----------------------------------------------------

        sql = """

        SELECT

            r.edge_id,

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
                sql,
                (
                    route_edges,
                )
            )

            rows = cur.fetchall()

            columns = [

                desc[0]

                for desc in cur.description

            ]


        edges = pd.DataFrame(
            rows,
            columns=columns
        )


        if edges.empty:

            raise RuntimeError(
                f"{route_name}: could not read route edges."
            )


        # ----------------------------------------------------
        # Preserve route order
        # ----------------------------------------------------

        order_map = {

            edge_id: position

            for position, edge_id

            in enumerate(route_edges)

        }


        edges["route_order"] = (

            edges["edge_id"]
            .map(order_map)

        )


        edges = (

            edges
            .sort_values("route_order")
            .reset_index(drop=True)

        )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        distance = float(
            edges["length_km"].sum()
        )


        dynamic_cost = float(
            edges["dynamic_weight"].sum()
        )


        avg_landslide = float(
            edges["p_landslide"].mean()
        )


        avg_flood = float(
            edges["p_flood"].mean()
        )


        avg_hazard = float(
            edges["p_hazard"].mean()
        )


        max_hazard = float(
            edges["p_hazard"].max()
        )


        high_segments = int(

            (
                edges["p_hazard"] >= 0.40
            ).sum()

        )


        very_high_segments = int(

            (
                edges["p_hazard"] >= 0.60
            ).sum()

        )


        extreme_segments = int(

            (
                edges["p_hazard"] >= 0.80
            ).sum()

        )


        return {

            "route": route_name,

            "edges": len(edges),

            "distance_km": distance,

            "dynamic_cost": dynamic_cost,

            "avg_landslide": avg_landslide,

            "avg_flood": avg_flood,

            "avg_hazard": avg_hazard,

            "max_hazard": max_hazard,

            "high_segments": high_segments,

            "very_high_segments": very_high_segments,

            "extreme_segments": extreme_segments

        }, edges


    # ========================================================
    # STEP 29C
    # SHORTEST DISTANCE ROUTE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "Calculating shortest-distance route..."
    )

    print(
        "=========================================="
    )


    shortest_route = run_route(
        "r.length_km"
    )


    shortest_stats, shortest_edges = (

        get_route_statistics(

            shortest_route,

            "Shortest route"

        )

    )


    print(
        "\nShortest route found."
    )


    print(
        f"Edges          : "
        f"{shortest_stats['edges']}"
    )


    print(
        f"Distance       : "
        f"{shortest_stats['distance_km']:.3f} km"
    )


    print(
        f"Dynamic cost   : "
        f"{shortest_stats['dynamic_cost']:.3f}"
    )


    print(
        f"Average hazard : "
        f"{shortest_stats['avg_hazard']:.4f}"
    )


    print(
        f"Maximum hazard : "
        f"{shortest_stats['max_hazard']:.4f}"
    )


    # ========================================================
    # STEP 29D
    # DYNAMIC RISK-AWARE ROUTE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "Calculating dynamic risk-aware route..."
    )

    print(
        "=========================================="
    )


    dynamic_route = run_route(
        "d.dynamic_weight"
    )


    dynamic_stats, dynamic_edges = (

        get_route_statistics(

            dynamic_route,

            "Dynamic risk-aware route"

        )

    )


    print(
        "\nDynamic route found."
    )


    print(
        f"Edges          : "
        f"{dynamic_stats['edges']}"
    )


    print(
        f"Distance       : "
        f"{dynamic_stats['distance_km']:.3f} km"
    )


    print(
        f"Dynamic cost   : "
        f"{dynamic_stats['dynamic_cost']:.3f}"
    )


    print(
        f"Average hazard : "
        f"{dynamic_stats['avg_hazard']:.4f}"
    )


    print(
        f"Maximum hazard : "
        f"{dynamic_stats['max_hazard']:.4f}"
    )


    # ========================================================
    # STEP 29E
    # COMPARE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "ROUTE COMPARISON"
    )

    print(
        "=========================================="
    )


    comparison = pd.DataFrame(

        [

            shortest_stats,

            dynamic_stats

        ]

    )


    print(

        comparison.to_string(
            index=False
        )

    )


    # ========================================================
    # CALCULATE IMPROVEMENTS
    # ========================================================

    distance_increase = (

        (
            dynamic_stats["distance_km"]

            -

            shortest_stats["distance_km"]

        )

        /

        shortest_stats["distance_km"]

    ) * 100


    hazard_reduction = (

        (
            shortest_stats["avg_hazard"]

            -

            dynamic_stats["avg_hazard"]

        )

        /

        shortest_stats["avg_hazard"]

    ) * 100


    maximum_hazard_reduction = (

        (
            shortest_stats["max_hazard"]

            -

            dynamic_stats["max_hazard"]

        )

        /

        shortest_stats["max_hazard"]

    ) * 100


    print(
        "\n=========================================="
    )

    print(
        "TRADE-OFF"
    )

    print(
        "=========================================="
    )


    print(
        f"Distance increase: "
        f"{distance_increase:.2f}%"
    )


    print(
        f"Average hazard reduction: "
        f"{hazard_reduction:.2f}%"
    )


    print(
        f"Maximum hazard reduction: "
        f"{maximum_hazard_reduction:.2f}%"
    )


    # ========================================================
    # HAZARD SEGMENT COMPARISON
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "HIGH-RISK SEGMENT COMPARISON"
    )

    print(
        "=========================================="
    )


    print(
        f"Shortest route:"
    )


    print(
        f"  Hazard >= 0.40 : "
        f"{shortest_stats['high_segments']}"
    )


    print(
        f"  Hazard >= 0.60 : "
        f"{shortest_stats['very_high_segments']}"
    )


    print(
        f"  Hazard >= 0.80 : "
        f"{shortest_stats['extreme_segments']}"
    )


    print(
        f"\nDynamic route:"
    )


    print(
        f"  Hazard >= 0.40 : "
        f"{dynamic_stats['high_segments']}"
    )


    print(
        f"  Hazard >= 0.60 : "
        f"{dynamic_stats['very_high_segments']}"
    )


    print(
        f"  Hazard >= 0.80 : "
        f"{dynamic_stats['extreme_segments']}"
    )


    # ========================================================
    # SAVE COMPARISON
    # ========================================================

    output_file = (

        r"C:\NERProject\Data\processed"
        r"\route_comparison.csv"

    )


    comparison.to_csv(

        output_file,

        index=False

    )


    print(
        "\nSaved comparison:"
    )


    print(
        output_file
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
    "STEP 29 COMPLETE"
)


print(
    "=========================================="
)


print(
    "\nShortest route and dynamic risk-aware "
    "route have been compared."
)


print(
    "\nNext step:"
)


print(
    "Step 30: Generate 6–7 km checkpoints "
    "along the selected route."
)

