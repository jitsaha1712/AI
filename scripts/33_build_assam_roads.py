import psycopg2


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "assam_routing",
    "user": "postgres",
    "password": "postgres",   # your local password
}


# ------------------------------------------------------------
# IMPORTANT
#
# This is a working Assam bounding box.
#
# It is intentionally a little larger than Assam so roads near
# the state boundary are not accidentally cut off.
#
# We will refine this with the actual Assam boundary later.
# ------------------------------------------------------------

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

    # --------------------------------------------------------
    # 1. Inspect source road table
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT COUNT(*)
        FROM road_edges;
        """
    )

    total_roads = cur.fetchone()[0]

    print(
        f"Total road_edges available: {total_roads:,}"
    )

    # --------------------------------------------------------
    # 2. Remove previous Assam working table
    # --------------------------------------------------------

    cur.execute(
        """
        DROP TABLE IF EXISTS routing_edges_assam;
        """
    )

    # --------------------------------------------------------
    # 3. Create Assam road subset
    #
    # ST_Intersects is used so roads crossing the bbox
    # boundary are retained.
    #
    # ST_Expand around the bbox is NOT required because
    # the bbox itself is deliberately generous.
    # --------------------------------------------------------

    cur.execute(
        """
        CREATE TABLE routing_edges_assam AS

        SELECT *
        FROM road_edges

        WHERE ST_Intersects(
            geometry,
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
            MAX_LAT,
        )
    )

    # --------------------------------------------------------
    # 4. Count Assam roads
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT COUNT(*)
        FROM routing_edges_assam;
        """
    )

    assam_roads = cur.fetchone()[0]

    print(
        f"Roads selected for Assam area: {assam_roads:,}"
    )

    if assam_roads == 0:

        raise RuntimeError(
            "No roads were found in the Assam bounding box."
        )

    # --------------------------------------------------------
    # 5. Spatial index
    # --------------------------------------------------------

    print(
        "\nCreating spatial index..."
    )

    cur.execute(
        """
        CREATE INDEX routing_edges_assam_geom_idx
        ON routing_edges_assam
        USING GIST (geometry);
        """
    )

    # --------------------------------------------------------
    # 6. Analyze
    # --------------------------------------------------------

    cur.execute(
        """
        ANALYZE routing_edges_assam;
        """
    )

    conn.commit()

    print(
        "\nSUCCESS: routing_edges_assam created."
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