from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR /
    "london_road_segment_risk_dataset.gpkg"
)

OUTPUT_CSV = (
    PROCESSED_DIR /
    "london_road_segment_risk_dataset.csv"
)


# ============================================================
# FIND ROAD NETWORK FILE
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - ROAD RISK DATASET")
print("=" * 80)

possible_files = [
    PROCESSED_DIR / "london_road_network.gpkg",
    PROCESSED_DIR / "london_road_segments.gpkg",
    PROCESSED_DIR / "london_roads.gpkg"
]

road_file = None

for file in possible_files:
    if file.exists():
        road_file = file
        break

if road_file is None:
    raise FileNotFoundError(
        "Road network GeoPackage could not be found.\n"
        "Check the output filename created by 04_road_network.py."
    )

print(f"\nRoad network file:\n{road_file}")


# ============================================================
# LOAD ROAD SEGMENTS
# ============================================================

roads = gpd.read_file(road_file)

print("\nRoad network loaded successfully.")

print(f"Road segments: {len(roads):,}")
print(f"CRS: {roads.crs}")
print(f"Columns: {len(roads.columns)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("ROAD DATA VALIDATION")
print("=" * 80)

if roads.crs is None:
    raise ValueError("Road network has no CRS.")

if roads.crs.to_epsg() != 27700:
    print("Transforming roads to EPSG:27700...")
    roads = roads.to_crs(27700)

print(
    f"Missing geometries: "
    f"{roads.geometry.isna().sum():,}"
)

print(
    f"Empty geometries: "
    f"{roads.geometry.is_empty.sum():,}"
)


roads = roads[
    roads.geometry.notna()
    &
    ~roads.geometry.is_empty
].copy()

roads = roads.reset_index(drop=True)


# ============================================================
# SEGMENT LENGTH
# ============================================================

print("\n" + "=" * 80)
print("ROAD LENGTH FEATURES")
print("=" * 80)

if "segment_length_m" not in roads.columns:
    roads["segment_length_m"] = roads.geometry.length

roads["segment_length_km"] = (
    roads["segment_length_m"] / 1000
)

print(
    roads["segment_length_m"]
    .describe()
    .round(2)
)


# ============================================================
# ENSURE COLLISION VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("COLLISION VARIABLES")
print("=" * 80)

collision_columns = [
    "collision_count",
    "severe_collision_count",
    "fatal_collision_count"
]

for column in collision_columns:

    if column not in roads.columns:
        print(
            f"{column} not found. "
            f"Creating with zeros."
        )

        roads[column] = 0

    roads[column] = (
        pd.to_numeric(
            roads[column],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


print(
    f"Total collisions represented: "
    f"{roads['collision_count'].sum():,}"
)

print(
    f"Total severe collisions represented: "
    f"{roads['severe_collision_count'].sum():,}"
)

print(
    f"Total fatal collisions represented: "
    f"{roads['fatal_collision_count'].sum():,}"
)


# ============================================================
# TARGET VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("TARGET VARIABLE CREATION")
print("=" * 80)

roads["has_collision"] = (
    roads["collision_count"] > 0
).astype(int)

roads["has_severe_collision"] = (
    roads["severe_collision_count"] > 0
).astype(int)


print("\nCollision target:")

print(
    roads[
        "has_collision"
    ]
    .value_counts()
)


print("\nSevere collision target:")

print(
    roads[
        "has_severe_collision"
    ]
    .value_counts()
)


# ============================================================
# COLLISION RATE
# ============================================================

roads["collisions_per_km"] = np.where(
    roads["segment_length_km"] > 0,
    roads["collision_count"]
    /
    roads["segment_length_km"],
    0
)

roads["severe_collisions_per_km"] = np.where(
    roads["segment_length_km"] > 0,
    roads["severe_collision_count"]
    /
    roads["segment_length_km"],
    0
)


# ============================================================
# ROAD TYPE CLEANING
# ============================================================

print("\n" + "=" * 80)
print("ROAD TYPE FEATURES")
print("=" * 80)

if "highway" in roads.columns:

    def clean_highway(value):

        if isinstance(value, list):
            return str(value[0])

        if pd.isna(value):
            return "unknown"

        value = str(value)

        if value.startswith("["):
            value = (
                value
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(",")[0]
                .strip()
            )

        return value


    roads["highway_clean"] = (
        roads["highway"]
        .apply(clean_highway)
    )

else:

    roads["highway_clean"] = "unknown"


print(
    roads[
        "highway_clean"
    ]
    .value_counts()
    .head(20)
)


# ============================================================
# ROAD HIERARCHY
# ============================================================

road_hierarchy = {
    "motorway": 6,
    "motorway_link": 6,

    "trunk": 5,
    "trunk_link": 5,

    "primary": 4,
    "primary_link": 4,

    "secondary": 3,
    "secondary_link": 3,

    "tertiary": 2,
    "tertiary_link": 2,

    "residential": 1,
    "living_street": 1,

    "unclassified": 1,
    "service": 0
}


roads["road_hierarchy"] = (
    roads[
        "highway_clean"
    ]
    .map(
        road_hierarchy
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# MAJOR ROAD FLAG
# ============================================================

major_types = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link"
]

roads["is_major_road"] = (
    roads[
        "highway_clean"
    ]
    .isin(
        major_types
    )
    .astype(int)
)


# ============================================================
# ROAD NAME FLAG
# ============================================================

if "name" in roads.columns:

    roads["has_road_name"] = (
        roads["name"]
        .notna()
        .astype(int)
    )

else:

    roads["has_road_name"] = 0


# ============================================================
# ONEWAY FEATURE
# ============================================================

print("\n" + "=" * 80)
print("ONEWAY FEATURE")
print("=" * 80)

if "oneway" in roads.columns:

    roads["is_oneway"] = (
        roads["oneway"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
        .astype(int)
    )

else:

    roads["is_oneway"] = 0


print(
    roads[
        "is_oneway"
    ]
    .value_counts()
)


# ============================================================
# BRIDGE FEATURE
# ============================================================

if "bridge" in roads.columns:

    roads["is_bridge"] = (
        roads["bridge"]
        .notna()
        .astype(int)
    )

else:

    roads["is_bridge"] = 0


# ============================================================
# TUNNEL FEATURE
# ============================================================

if "tunnel" in roads.columns:

    roads["is_tunnel"] = (
        roads["tunnel"]
        .notna()
        .astype(int)
    )

else:

    roads["is_tunnel"] = 0


# ============================================================
# SPEED INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("SPEED INFORMATION")
print("=" * 80)

if "maxspeed" in roads.columns:

    roads["maxspeed_numeric"] = (
        roads["maxspeed"]
        .astype(str)
        .str.extract(
            r"(\d+)"
        )[0]
    )

    roads["maxspeed_numeric"] = (
        pd.to_numeric(
            roads[
                "maxspeed_numeric"
            ],
            errors="coerce"
        )
    )

else:

    roads["maxspeed_numeric"] = np.nan


print(
    f"Road segments with explicit maxspeed: "
    f"{roads['maxspeed_numeric'].notna().sum():,}"
)


# ============================================================
# GEOMETRIC FEATURES
# ============================================================

print("\n" + "=" * 80)
print("GEOMETRIC FEATURES")
print("=" * 80)

roads["centroid_x"] = (
    roads.geometry.centroid.x
)

roads["centroid_y"] = (
    roads.geometry.centroid.y
)


# ============================================================
# COLLISION TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

total = len(roads)

positive = (
    roads[
        "has_collision"
    ]
    .sum()
)

negative = (
    total -
    positive
)

positive_rate = (
    positive /
    total *
    100
)


print(
    f"Total road segments: "
    f"{total:,}"
)

print(
    f"Segments with collision: "
    f"{positive:,}"
)

print(
    f"Segments without collision: "
    f"{negative:,}"
)

print(
    f"Positive class rate: "
    f"{positive_rate:.2f}%"
)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATA QUALITY CHECK")
print("=" * 80)

model_features = [
    "segment_length_m",
    "road_hierarchy",
    "is_major_road",
    "has_road_name",
    "is_oneway",
    "is_bridge",
    "is_tunnel",
    "centroid_x",
    "centroid_y"
]


for feature in model_features:

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
# SAVE GEOPACKAGE
# ============================================================

print("\n" + "=" * 80)
print("SAVE ROAD RISK DATASET")
print("=" * 80)

roads.to_file(
    OUTPUT_FILE,
    layer="road_risk_dataset",
    driver="GPKG"
)

print(
    f"GeoPackage saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# SAVE CSV
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
print("ROAD RISK DATASET SUMMARY")
print("=" * 80)

print(
    f"Road segments: "
    f"{len(roads):,}"
)

print(
    f"Segments with collisions: "
    f"{roads['has_collision'].sum():,}"
)

print(
    f"Segments with severe collisions: "
    f"{roads['has_severe_collision'].sum():,}"
)

print(
    f"Collision rate: "
    f"{roads['has_collision'].mean() * 100:.2f}%"
)

print(
    f"Severe collision rate: "
    f"{roads['has_severe_collision'].mean() * 100:.2f}%"
)


print("\n" + "=" * 80)
print("ROAD SEGMENT RISK DATASET COMPLETED")
print("=" * 80)