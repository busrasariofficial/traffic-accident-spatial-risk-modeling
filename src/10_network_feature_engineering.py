from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

INPUT_ROADS = (
    PROCESSED_DIR /
    "london_road_segment_risk_dataset.gpkg"
)

OUTPUT_GPKG = (
    PROCESSED_DIR /
    "london_road_segment_network_features.gpkg"
)

OUTPUT_CSV = (
    PROCESSED_DIR /
    "london_road_segment_network_features.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - NETWORK FEATURE ENGINEERING")
print("=" * 80)


# ============================================================
# 1. LOAD ROAD SEGMENTS
# ============================================================

if not INPUT_ROADS.exists():

    raise FileNotFoundError(
        f"Road risk dataset not found:\n"
        f"{INPUT_ROADS}"
    )


roads = gpd.read_file(
    INPUT_ROADS,
    layer="road_risk_dataset"
)


print("\nRoad risk dataset loaded successfully.")

print(
    f"\nRoad segments: "
    f"{len(roads):,}"
)

print(
    f"CRS: "
    f"{roads.crs}"
)


# ============================================================
# 2. CRS VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("CRS VALIDATION")
print("=" * 80)


if roads.crs is None:

    raise ValueError(
        "Road dataset has no CRS."
    )


if roads.crs.to_epsg() != 27700:

    print(
        "Transforming road dataset to EPSG:27700..."
    )

    roads = roads.to_crs(
        27700
    )


print(
    f"Analysis CRS: "
    f"{roads.crs}"
)


# ============================================================
# 3. GEOMETRY VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("GEOMETRY VALIDATION")
print("=" * 80)


print(
    f"Missing geometries: "
    f"{roads.geometry.isna().sum():,}"
)

print(
    f"Empty geometries: "
    f"{roads.geometry.is_empty.sum():,}"
)


roads = (
    roads[
        roads.geometry.notna()
        &
        ~roads.geometry.is_empty
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    f"Valid road segments: "
    f"{len(roads):,}"
)


# ============================================================
# 4. EXTRACT ROAD ENDPOINTS
# ============================================================

print("\n" + "=" * 80)
print("ROAD ENDPOINT EXTRACTION")
print("=" * 80)


def get_start_point(geometry):

    if geometry is None:
        return None

    if geometry.geom_type == "LineString":

        return geometry.coords[0]

    if geometry.geom_type == "MultiLineString":

        first_line = list(
            geometry.geoms
        )[0]

        return first_line.coords[0]

    return None


def get_end_point(geometry):

    if geometry is None:
        return None

    if geometry.geom_type == "LineString":

        return geometry.coords[-1]

    if geometry.geom_type == "MultiLineString":

        last_line = list(
            geometry.geoms
        )[-1]

        return last_line.coords[-1]

    return None


start_coordinates = (
    roads.geometry
    .apply(
        get_start_point
    )
)

end_coordinates = (
    roads.geometry
    .apply(
        get_end_point
    )
)


roads["start_x"] = [
    coordinate[0]
    if coordinate is not None
    else np.nan
    for coordinate in start_coordinates
]

roads["start_y"] = [
    coordinate[1]
    if coordinate is not None
    else np.nan
    for coordinate in start_coordinates
]

roads["end_x"] = [
    coordinate[0]
    if coordinate is not None
    else np.nan
    for coordinate in end_coordinates
]

roads["end_y"] = [
    coordinate[1]
    if coordinate is not None
    else np.nan
    for coordinate in end_coordinates
]


print(
    f"Valid start coordinates: "
    f"{roads['start_x'].notna().sum():,}"
)

print(
    f"Valid end coordinates: "
    f"{roads['end_x'].notna().sum():,}"
)


# ============================================================
# 5. CREATE APPROXIMATE NODE IDS
# ============================================================

print("\n" + "=" * 80)
print("NETWORK NODE CREATION")
print("=" * 80)

# Coordinates are rounded to the nearest metre.
#
# This allows endpoints occupying effectively the same
# physical location to be treated as the same network node.

roads["start_node"] = (
    roads["start_x"]
    .round(0)
    .astype("Int64")
    .astype(str)
    +
    "_"
    +
    roads["start_y"]
    .round(0)
    .astype("Int64")
    .astype(str)
)


roads["end_node"] = (
    roads["end_x"]
    .round(0)
    .astype("Int64")
    .astype(str)
    +
    "_"
    +
    roads["end_y"]
    .round(0)
    .astype("Int64")
    .astype(str)
)


unique_nodes = pd.unique(
    pd.concat(
        [
            roads["start_node"],
            roads["end_node"]
        ],
        ignore_index=True
    )
)


print(
    f"Approximate unique network nodes: "
    f"{len(unique_nodes):,}"
)


# ============================================================
# 6. NODE DEGREE
# ============================================================

print("\n" + "=" * 80)
print("NODE DEGREE CALCULATION")
print("=" * 80)


all_nodes = pd.concat(
    [
        roads[
            ["start_node"]
        ]
        .rename(
            columns={
                "start_node": "node"
            }
        ),

        roads[
            ["end_node"]
        ]
        .rename(
            columns={
                "end_node": "node"
            }
        )
    ],
    ignore_index=True
)


node_degree = (
    all_nodes[
        "node"
    ]
    .value_counts()
)


roads["start_degree"] = (
    roads[
        "start_node"
    ]
    .map(
        node_degree
    )
    .fillna(1)
    .astype(int)
)


roads["end_degree"] = (
    roads[
        "end_node"
    ]
    .map(
        node_degree
    )
    .fillna(1)
    .astype(int)
)


print("\nStart-node degree:")

print(
    roads[
        "start_degree"
    ]
    .describe()
    .round(2)
)


print("\nEnd-node degree:")

print(
    roads[
        "end_degree"
    ]
    .describe()
    .round(2)
)


# ============================================================
# 7. SEGMENT CONNECTIVITY
# ============================================================

print("\n" + "=" * 80)
print("SEGMENT CONNECTIVITY FEATURES")
print("=" * 80)


roads["max_node_degree"] = (
    roads[
        [
            "start_degree",
            "end_degree"
        ]
    ]
    .max(
        axis=1
    )
)


roads["mean_node_degree"] = (
    roads[
        [
            "start_degree",
            "end_degree"
        ]
    ]
    .mean(
        axis=1
    )
)


roads["degree_difference"] = (
    (
        roads[
            "start_degree"
        ]
        -
        roads[
            "end_degree"
        ]
    )
    .abs()
)


# ============================================================
# 8. INTERSECTION FEATURES
# ============================================================

print("\n" + "=" * 80)
print("INTERSECTION FEATURES")
print("=" * 80)

# degree >= 3 is used as an approximation of an
# intersection / network junction.

roads["start_is_intersection"] = (
    roads[
        "start_degree"
    ] >= 3
).astype(int)


roads["end_is_intersection"] = (
    roads[
        "end_degree"
    ] >= 3
).astype(int)


roads["touches_intersection"] = (
    (
        roads[
            "start_is_intersection"
        ] == 1
    )
    |
    (
        roads[
            "end_is_intersection"
        ] == 1
    )
).astype(int)


roads["intersection_end_count"] = (
    roads[
        "start_is_intersection"
    ]
    +
    roads[
        "end_is_intersection"
    ]
)


print(
    "Segments touching an intersection: "
    f"{roads['touches_intersection'].sum():,}"
)


print(
    "\nIntersection endpoint count:"
)

print(
    roads[
        "intersection_end_count"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. COMPLEX INTERSECTION FLAG
# ============================================================

print("\n" + "=" * 80)
print("COMPLEX INTERSECTION FEATURES")
print("=" * 80)

# degree >= 4 represents a more connected / complex node.

roads["touches_complex_intersection"] = (
    roads[
        "max_node_degree"
    ] >= 4
).astype(int)


print(
    "Segments touching complex intersection: "
    f"{roads['touches_complex_intersection'].sum():,}"
)


# ============================================================
# 10. DEAD-END FEATURES
# ============================================================

print("\n" + "=" * 80)
print("DEAD-END FEATURES")
print("=" * 80)


roads["start_dead_end"] = (
    roads[
        "start_degree"
    ] == 1
).astype(int)


roads["end_dead_end"] = (
    roads[
        "end_degree"
    ] == 1
).astype(int)


roads["touches_dead_end"] = (
    (
        roads[
            "start_dead_end"
        ] == 1
    )
    |
    (
        roads[
            "end_dead_end"
        ] == 1
    )
).astype(int)


print(
    f"Segments touching dead ends: "
    f"{roads['touches_dead_end'].sum():,}"
)


# ============================================================
# 11. ROAD LENGTH TRANSFORMATION
# ============================================================

print("\n" + "=" * 80)
print("ROAD LENGTH TRANSFORMATION")
print("=" * 80)


roads["log_segment_length"] = np.log1p(
    roads[
        "segment_length_m"
    ]
)


print(
    roads[
        [
            "segment_length_m",
            "log_segment_length"
        ]
    ]
    .describe()
    .round(3)
)


# ============================================================
# 12. ROAD SPEED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("ROAD SPEED FEATURES")
print("=" * 80)


if "maxspeed_numeric" in roads.columns:

    median_speed = (
        roads[
            "maxspeed_numeric"
        ]
        .median()
    )


    print(
        f"Median explicit speed: "
        f"{median_speed:.2f}"
    )


    roads[
        "maxspeed_missing"
    ] = (
        roads[
            "maxspeed_numeric"
        ]
        .isna()
        .astype(int)
    )


    roads[
        "high_speed_osm"
    ] = (
        roads[
            "maxspeed_numeric"
        ] >= 50
    ).astype(int)


else:

    roads[
        "maxspeed_missing"
    ] = 1

    roads[
        "high_speed_osm"
    ] = 0


print(
    f"Segments missing maxspeed: "
    f"{roads['maxspeed_missing'].sum():,}"
)


print(
    f"Explicit high-speed segments: "
    f"{roads['high_speed_osm'].sum():,}"
)


# ============================================================
# 13. ROAD COMPLEXITY INDEX
# ============================================================

print("\n" + "=" * 80)
print("ROAD COMPLEXITY INDEX")
print("=" * 80)

# Simple interpretable engineered indicator.
#
# It combines:
#
# - network connectivity
# - intersection exposure
# - road hierarchy
# - one-way status

roads["road_complexity_index"] = (
    roads[
        "mean_node_degree"
    ]
    +
    roads[
        "touches_intersection"
    ]
    +
    roads[
        "touches_complex_intersection"
    ]
    +
    (
        roads[
            "road_hierarchy"
        ] * 0.5
    )
    +
    (
        roads[
            "is_oneway"
        ] * 0.5
    )
)


print(
    roads[
        "road_complexity_index"
    ]
    .describe()
    .round(3)
)


# ============================================================
# 14. TARGET COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("NETWORK FEATURES VS COLLISION TARGET")
print("=" * 80)


comparison_columns = [
    "segment_length_m",
    "mean_node_degree",
    "max_node_degree",
    "touches_intersection",
    "touches_complex_intersection",
    "road_hierarchy",
    "is_major_road",
    "is_oneway",
    "road_complexity_index"
]


comparison = (
    roads
    .groupby(
        "has_collision"
    )[
        comparison_columns
    ]
    .mean()
    .T
)


comparison.columns = [
    "No_Collision",
    "Collision"
]


comparison[
    "Difference"
] = (
    comparison[
        "Collision"
    ]
    -
    comparison[
        "No_Collision"
    ]
)


print(
    comparison
    .round(3)
    .to_string()
)


# ============================================================
# 15. FEATURE QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE QUALITY CHECK")
print("=" * 80)


new_features = [
    "start_degree",
    "end_degree",
    "max_node_degree",
    "mean_node_degree",
    "degree_difference",
    "start_is_intersection",
    "end_is_intersection",
    "touches_intersection",
    "intersection_end_count",
    "touches_complex_intersection",
    "touches_dead_end",
    "log_segment_length",
    "maxspeed_missing",
    "high_speed_osm",
    "road_complexity_index"
]


for feature in new_features:

    missing = (
        roads[
            feature
        ]
        .isna()
        .sum()
    )

    print(
        f"{feature}: "
        f"{missing:,} missing"
    )


# ============================================================
# 16. SAVE GEOPACKAGE
# ============================================================

print("\n" + "=" * 80)
print("SAVE NETWORK FEATURE DATASET")
print("=" * 80)


roads.to_file(
    OUTPUT_GPKG,
    layer="network_features",
    driver="GPKG"
)


print(
    f"GeoPackage saved to:\n"
    f"{OUTPUT_GPKG}"
)


# ============================================================
# 17. SAVE CSV
# ============================================================

csv_df = (
    roads
    .drop(
        columns="geometry"
    )
    .copy()
)


csv_df.to_csv(
    OUTPUT_CSV,
    index=False
)


print(
    f"\nCSV saved to:\n"
    f"{OUTPUT_CSV}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("NETWORK FEATURE ENGINEERING SUMMARY")
print("=" * 80)


print(
    f"Road segments: "
    f"{len(roads):,}"
)

print(
    f"Approximate network nodes: "
    f"{len(unique_nodes):,}"
)

print(
    f"Segments touching intersections: "
    f"{roads['touches_intersection'].sum():,}"
)

print(
    f"Segments touching complex intersections: "
    f"{roads['touches_complex_intersection'].sum():,}"
)

print(
    f"Segments touching dead ends: "
    f"{roads['touches_dead_end'].sum():,}"
)

print(
    f"New network features created: "
    f"{len(new_features)}"
)


print("\n" + "=" * 80)
print("NETWORK FEATURE ENGINEERING COMPLETED")
print("=" * 80)