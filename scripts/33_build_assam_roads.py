import psycopg2


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "assam_routing",
    "user": "postgres",
    "password": "postgres",   # PUT YOUR LOCAL POSTGRES PASSWORD HERE
}


# ============================================================
# ASSAM AREA
# ============================================================

MIN_LON = 89.5
MAX_LON = 96.2

MIN_LAT = 24.0
MAX_LAT = 28.3


conn = None


try:

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    conn.autocommit = False

    cur = conn.cursor()

    print("\nConnected to PostgreSQL.")


    # ========================================================
    # 1. BASIC TABLE CHECK
    # ========================================================

    cur.execute(
        """
        SELECT COUNT(*)
        FROM public.road_edges
        WHERE geometry IS NOT NULL;
        """
    )

    total_roads = cur.fetchone()[0]

    print(
        f"Roads with geometry: {total_roads:,}"
    )


    # ========================================================
    # 2. CHECK CRS
    # ========================================================

    cur.execute(
        """
        SELECT
            ST_SRID(geometry)
        FROM public.road_edges
        WHERE geometry IS NOT NULL
        LIMIT 1;
        """
    )

    srid = cur.fetchone()[0]

    print(
        f"Road geometry SRID: EPSG:{srid}"
    )


    # ========================================================
    # 3. TEST ASSAM BBOX
    #
    # Transform each road FROM UTM 45N -> WGS84
    # and test against the WGS84 Assam bounding box.
    # ========================================================

    print(
        "\nChecking Assam road coverage..."
    )

    cur.execute(
        """
        SELECT COUNT(*)

        FROM public.road_edges

        WHERE geometry IS NOT NULL

        AND ST_Intersects(

            ST_Transform(
                geometry,
                4326
            ),

            ST_MakeEnvelope(
                %s,
                %s,
                %s,
                %s,
                4326
            )

        );
        """,
        (
            MIN_LON,
            MIN_LAT,
            MAX_LON,
            MAX_LAT
        )
    )

    selected_count = cur.fetchone()[0]

    print(
        f"Roads inside Assam bbox: {selected_count:,}"
    )


    if selected_count == 0:

        raise RuntimeError(
            "Still zero roads. Run the diagnostic SQL queries next."
        )


    # ========================================================
    # 4. DROP OLD TABLE
    # ========================================================

    cur.execute(
        """
        DROP TABLE IF EXISTS
        public.routing_edges_assam;
        """
    )


    # ========================================================
    # 5. CREATE ASSAM ROAD TABLE
    # ========================================================

    print(
        "\nCreating routing_edges_assam..."
    )

    cur.execute(
        """
        CREATE TABLE public.routing_edges_assam AS

        SELECT *

        FROM public.road_edges

        WHERE geometry IS NOT NULL

        AND ST_Intersects(

            ST_Transform(
                geometry,
                4326
            ),

            ST_MakeEnvelope(
                %s,
                %s,
                %s,
                %s,
                4326
            )

        );
        """,
        (
            MIN_LON,
            MIN_LAT,
            MAX_LON,
            MAX_LAT
        )
    )


    # ========================================================
    # 6. CREATE SPATIAL INDEX
    # ========================================================

    print(
        "Creating spatial index..."
    )

    cur.execute(
        """
        CREATE INDEX
        routing_edges_assam_geom_idx

        ON public.routing_edges_assam

        USING GIST (
            geometry
        );
        """
    )


    # ========================================================
    # 7. ANALYZE
    # ========================================================

    cur.execute(
        """
        ANALYZE public.routing_edges_assam;
        """
    )


    # ========================================================
    # 8. FINAL COUNT
    # ========================================================

    cur.execute(
        """
        SELECT COUNT(*)
        FROM public.routing_edges_assam;
        """
    )

    final_count = cur.fetchone()[0]

    print(
        f"\nFinal Assam road count: {final_count:,}"
    )


    # ========================================================
    # 9. GEOGRAPHIC EXTENT
    # ========================================================

    cur.execute(
        """
        SELECT

            ST_XMin(
                ST_Extent(
                    ST_Transform(
                        geometry,
                        4326
                    )
                )
            ) AS min_lon,

            ST_YMin(
                ST_Extent(
                    ST_Transform(
                        geometry,
                        4326
                    )
                )
            ) AS min_lat,

            ST_XMax(
                ST_Extent(
                    ST_Transform(
                        geometry,
                        4326
                    )
                )
            ) AS max_lon,

            ST_YMax(
                ST_Extent(
                    ST_Transform(
                        geometry,
                        4326
                    )
                )
            ) AS max_lat

        FROM public.routing_edges_assam;
        """
    )

    extent = cur.fetchone()

    print("\nAssam road extent:")
    print(
        f"Longitude: {extent[0]:.6f} -> {extent[2]:.6f}"
    )
    print(
        f"Latitude : {extent[1]:.6f} -> {extent[3]:.6f}"
    )


    # ========================================================
    # 10. COMMIT
    # ========================================================

    conn.commit()

    print(
        "\n========================================"
    )

    print(
        "SUCCESS!"
    )

    print(
        "Created: public.routing_edges_assam"
    )

    print(
        "========================================"
    )


except Exception as e:

    if conn is not None:

        conn.rollback()

    print(
        "\nERROR:",
        str(e)
    )


finally:

    if conn is not None:

        conn.close()

    print(
        "\nDone."
    )