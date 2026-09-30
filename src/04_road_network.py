from pathlib import Path

import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_GPKG = (
    PROCESSED_DATA_DIR /
    "traffic_collisions_2025_spatial.gpkg"
)

OUTPUT_COLLISIONS = (
    PROCESSED_DATA_DIR /
    "london_collisions_2025.gpkg"
)

OUTPUT_ROADS = (
    PROCESSED_DATA_DIR /
    "london_road_network.gpkg"
)

OUTPUT_MATCHED = (
    PROCESSED_DATA_DIR /
    "london_collisions_road_matched.gpkg"
)

OUTPUT_MATCHED_CSV = (
    PROCESSED_DATA_DIR /
    "london_collisions_road_matched.csv"
)


# ============================================================
# SETTINGS
# ============================================================

PLACE_NAME = "London, England, United Kingdom"

# Road network type:
# drive = roads usable by motor vehicles
NETWORK_TYPE = "drive"


# ============================================================
# LOAD COLLISION DATA
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - ROAD NETWORK")
print("=" * 80)

if not INPUT_GPKG.exists():
    raise FileNotFoundError(
        f"Spatial dataset not found:\n{INPUT_GPKG}"
    )

gdf = gpd.read_file(
    INPUT_GPKG,
    layer="collisions_2025"
)

print("\nSpatial collision dataset loaded successfully.")

print(f"\nTotal collisions: {len(gdf):,}")
print(f"Collision CRS: {gdf.crs}")


# ============================================================
# 1. DOWNLOAD LONDON BOUNDARY
# ============================================================

print("\n" + "=" * 80)
print("LONDON BOUNDARY")
print("=" * 80)

print(
    f"Downloading administrative boundary for:\n"
    f"{PLACE_NAME}"
)

london_boundary = ox.geocode_to_gdf(
    PLACE_NAME
)

print("\nLondon boundary downloaded successfully.")

print(f"Boundary CRS: {london_boundary.crs}")


# ============================================================
# 2. PROJECT BOUNDARY
# ============================================================

london_boundary = london_boundary.to_crs(
    gdf.crs
)

print(
    f"Projected boundary CRS: "
    f"{london_boundary.crs}"
)


# ============================================================
# 3. SELECT LONDON COLLISIONS
# ============================================================

print("\n" + "=" * 80)
print("LONDON COLLISION SELECTION")
print("=" * 80)

london_collisions = gpd.sjoin(
    gdf,
    london_boundary[["geometry"]],
    how="inner",
    predicate="within"
)

# Remove spatial join helper column
london_collisions = (
    london_collisions
    .drop(
        columns=["index_right"],
        errors="ignore"
    )
    .copy()
)

print(
    f"London collisions selected: "
    f"{len(london_collisions):,}"
)

if london_collisions.empty:
    raise ValueError(
        "No collisions were found inside the London boundary."
    )


# ============================================================
# 4. LONDON COLLISION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("LONDON COLLISION SUMMARY")
print("=" * 80)

print("\nSeverity:")

print(
    london_collisions[
        "severity_label"
    ]
    .value_counts()
)

print("\nUrban / rural:")

print(
    london_collisions[
        "urban_rural_label"
    ]
    .value_counts()
)

print("\nSpeed limits:")

print(
    london_collisions[
        "speed_limit"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. DOWNLOAD ROAD NETWORK
# ============================================================

print("\n" + "=" * 80)
print("ROAD NETWORK DOWNLOAD")
print("=" * 80)

print(
    "Downloading London drivable road network..."
)

print(
    "This can take several minutes depending on "
    "internet connection and OpenStreetMap response time."
)

G = ox.graph_from_place(
    PLACE_NAME,
    network_type=NETWORK_TYPE,
    simplify=True
)

print("\nRoad network downloaded successfully.")

print(
    f"Network nodes: "
    f"{len(G.nodes):,}"
)

print(
    f"Network edges: "
    f"{len(G.edges):,}"
)


# ============================================================
# 6. PROJECT ROAD NETWORK
# ============================================================

print("\n" + "=" * 80)
print("ROAD NETWORK PROJECTION")
print("=" * 80)

# Project directly to British National Grid
# so collisions and road geometries share the same CRS.

G_projected = ox.project_graph(
    G,
    to_crs="EPSG:27700"
)

print(
    f"Projected network CRS: "
    f"{G_projected.graph['crs']}"
)


# ============================================================
# 7. CONVERT NETWORK TO GEODATAFRAMES
# ============================================================

print("\n" + "=" * 80)
print("ROAD NETWORK GEODATAFRAMES")
print("=" * 80)

nodes, edges = ox.graph_to_gdfs(
    G_projected
)

print(
    f"Road nodes: "
    f"{len(nodes):,}"
)

print(
    f"Road segments: "
    f"{len(edges):,}"
)


# ============================================================
# 8. ROAD SEGMENT LENGTH QUALITY
# ============================================================

print("\n" + "=" * 80)
print("ROAD SEGMENT LENGTH QUALITY")
print("=" * 80)

edges["segment_length_m"] = (
    edges.geometry.length
)

print(
    edges["segment_length_m"]
    .describe()
    .round(2)
)

zero_length = (
    edges["segment_length_m"] <= 0
).sum()

print(
    f"\nZero/non-positive length segments: "
    f"{zero_length:,}"
)


# ============================================================
# 9. PREPARE COLLISION COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("COLLISION COORDINATES")
print("=" * 80)

london_collisions = london_collisions.to_crs(
    "EPSG:27700"
)

collision_x = (
    london_collisions.geometry.x
    .to_numpy()
)

collision_y = (
    london_collisions.geometry.y
    .to_numpy()
)

print(
    f"Collision points prepared: "
    f"{len(london_collisions):,}"
)


# ============================================================
# 10. MATCH COLLISIONS TO NEAREST ROAD SEGMENT
# ============================================================

print("\n" + "=" * 80)
print("NEAREST ROAD SEGMENT MATCHING")
print("=" * 80)

print(
    "Matching each collision to its nearest "
    "OpenStreetMap road segment..."
)

nearest_edges = ox.distance.nearest_edges(
    G_projected,
    X=collision_x,
    Y=collision_y
)

# nearest_edges returns:
# (u, v, key)

london_collisions["road_u"] = [
    edge[0]
    for edge in nearest_edges
]

london_collisions["road_v"] = [
    edge[1]
    for edge in nearest_edges
]

london_collisions["road_key"] = [
    edge[2]
    for edge in nearest_edges
]

print(
    f"Matched collisions: "
    f"{len(london_collisions):,}"
)


# ============================================================
# 11. CREATE ROAD SEGMENT ID
# ============================================================

print("\n" + "=" * 80)
print("ROAD SEGMENT IDENTIFIERS")
print("=" * 80)

london_collisions["road_segment_id"] = (
    london_collisions["road_u"].astype(str)
    + "_"
    + london_collisions["road_v"].astype(str)
    + "_"
    + london_collisions["road_key"].astype(str)
)

edges = edges.reset_index()

edges["road_segment_id"] = (
    edges["u"].astype(str)
    + "_"
    + edges["v"].astype(str)
    + "_"
    + edges["key"].astype(str)
)

print(
    f"Unique road segments with collisions: "
    f"{london_collisions['road_segment_id'].nunique():,}"
)


# ============================================================
# 12. DISTANCE FROM COLLISION TO MATCHED ROAD
# ============================================================

print("\n" + "=" * 80)
print("COLLISION-TO-ROAD DISTANCE")
print("=" * 80)

road_geometry_lookup = (
    edges
    .set_index("road_segment_id")[
        "geometry"
    ]
)

matched_road_geometry = (
    london_collisions[
        "road_segment_id"
    ]
    .map(road_geometry_lookup)
)

london_collisions[
    "distance_to_road_m"
] = [
    collision_geometry.distance(
        road_geometry
    )
    if road_geometry is not None
    else np.nan

    for collision_geometry, road_geometry
    in zip(
        london_collisions.geometry,
        matched_road_geometry
    )
]

print(
    london_collisions[
        "distance_to_road_m"
    ]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .round(2)
)

print(
    f"\nCollisions within 10 m of matched road: "
    f"{(london_collisions['distance_to_road_m'] <= 10).sum():,}"
)

print(
    f"Collisions within 25 m of matched road: "
    f"{(london_collisions['distance_to_road_m'] <= 25).sum():,}"
)

print(
    f"Collisions within 50 m of matched road: "
    f"{(london_collisions['distance_to_road_m'] <= 50).sum():,}"
)


# ============================================================
# 13. COLLISION COUNT PER ROAD SEGMENT
# ============================================================

print("\n" + "=" * 80)
print("COLLISIONS PER ROAD SEGMENT")
print("=" * 80)

segment_collision_counts = (
    london_collisions
    .groupby(
        "road_segment_id"
    )
    .agg(
        collision_count=(
            "collision_index",
            "count"
        ),
        severe_collision_count=(
            "is_severe",
            "sum"
        ),
        fatal_collision_count=(
            "severity_label",
            lambda x: (
                x == "Fatal"
            ).sum()
        )
    )
    .reset_index()
)

segment_collision_counts[
    "severe_collision_rate"
] = (
    segment_collision_counts[
        "severe_collision_count"
    ]
    /
    segment_collision_counts[
        "collision_count"
    ]
)

print(
    f"Road segments with at least one collision: "
    f"{len(segment_collision_counts):,}"
)

print("\nCollision count per affected road segment:")

print(
    segment_collision_counts[
        "collision_count"
    ]
    .describe()
    .round(2)
)


# ============================================================
# 14. ADD COLLISION METRICS TO ROAD NETWORK
# ============================================================

print("\n" + "=" * 80)
print("ROAD SEGMENT COLLISION METRICS")
print("=" * 80)

edges = edges.merge(
    segment_collision_counts,
    on="road_segment_id",
    how="left"
)

count_columns = [
    "collision_count",
    "severe_collision_count",
    "fatal_collision_count"
]

edges[count_columns] = (
    edges[count_columns]
    .fillna(0)
)

edges[count_columns] = (
    edges[count_columns]
    .astype(int)
)

edges["has_collision"] = (
    edges["collision_count"] > 0
).astype(int)


# ============================================================
# 15. COLLISION DENSITY PER KM
# ============================================================

edges["segment_length_km"] = (
    edges["segment_length_m"] / 1000
)

edges[
    "collisions_per_km"
] = np.where(
    edges["segment_length_km"] > 0,

    edges["collision_count"]
    /
    edges["segment_length_km"],

    np.nan
)


# ============================================================
# 16. TOP COLLISION ROAD SEGMENTS
# ============================================================

print("\n" + "=" * 80)
print("TOP COLLISION ROAD SEGMENTS")
print("=" * 80)

top_segments = (
    edges[
        edges["collision_count"] > 0
    ]
    .sort_values(
        "collision_count",
        ascending=False
    )
    [
        [
            "road_segment_id",
            "name",
            "highway",
            "segment_length_m",
            "collision_count",
            "severe_collision_count",
            "fatal_collision_count",
            "collisions_per_km"
        ]
    ]
    .head(20)
)

print(
    top_segments.to_string(
        index=False
    )
)


# ============================================================
# 17. ROAD NETWORK SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ROAD NETWORK SUMMARY")
print("=" * 80)

print(
    f"Total road segments: "
    f"{len(edges):,}"
)

print(
    f"Segments with collisions: "
    f"{edges['has_collision'].sum():,}"
)

print(
    f"Segments without collisions: "
    f"{(edges['has_collision'] == 0).sum():,}"
)

print(
    f"Total matched collisions: "
    f"{int(edges['collision_count'].sum()):,}"
)

print(
    f"Total severe collisions: "
    f"{int(edges['severe_collision_count'].sum()):,}"
)

print(
    f"Total fatal collisions: "
    f"{int(edges['fatal_collision_count'].sum()):,}"
)


# ============================================================
# 18. SAVE LONDON COLLISIONS
# ============================================================

print("\n" + "=" * 80)
print("SAVE ROAD NETWORK OUTPUTS")
print("=" * 80)

london_collisions.to_file(
    OUTPUT_COLLISIONS,
    layer="london_collisions_2025",
    driver="GPKG"
)

print(
    f"London collisions saved to:\n"
    f"{OUTPUT_COLLISIONS}"
)


# ============================================================
# 19. SAVE ROAD NETWORK
# ============================================================

edges_gdf = gpd.GeoDataFrame(
    edges,
    geometry="geometry",
    crs="EPSG:27700"
)

# Some OSM attributes can contain Python lists.
# Convert them to strings for reliable GeoPackage export.

for column in edges_gdf.columns:

    if column == "geometry":
        continue

    has_list = edges_gdf[column].apply(
        lambda value: isinstance(
            value,
            (list, tuple, set)
        )
    ).any()

    if has_list:
        edges_gdf[column] = (
            edges_gdf[column]
            .astype(str)
        )


edges_gdf.to_file(
    OUTPUT_ROADS,
    layer="london_roads",
    driver="GPKG"
)

print(
    f"\nLondon road network saved to:\n"
    f"{OUTPUT_ROADS}"
)


# ============================================================
# 20. SAVE MATCHED COLLISIONS
# ============================================================

london_collisions.to_file(
    OUTPUT_MATCHED,
    layer="collision_road_matches",
    driver="GPKG"
)

print(
    f"\nRoad-matched collisions saved to:\n"
    f"{OUTPUT_MATCHED}"
)


# ============================================================
# 21. SAVE MATCHED CSV
# ============================================================

matched_csv = (
    london_collisions
    .drop(columns="geometry")
    .copy()
)

matched_csv.to_csv(
    OUTPUT_MATCHED_CSV,
    index=False
)

print(
    f"\nRoad-matched CSV saved to:\n"
    f"{OUTPUT_MATCHED_CSV}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL ROAD NETWORK CHECK")
print("=" * 80)

print(
    f"London collisions: "
    f"{len(london_collisions):,}"
)

print(
    f"Road segments: "
    f"{len(edges_gdf):,}"
)

print(
    f"Unique matched road segments: "
    f"{london_collisions['road_segment_id'].nunique():,}"
)

print(
    f"Missing road matches: "
    f"{london_collisions['road_segment_id'].isna().sum():,}"
)

print(
    f"Median collision-to-road distance: "
    f"{london_collisions['distance_to_road_m'].median():.2f} m"
)

print(
    f"95th percentile collision-to-road distance: "
    f"{london_collisions['distance_to_road_m'].quantile(0.95):.2f} m"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("ROAD NETWORK ANALYSIS COMPLETED")
print("=" * 80)