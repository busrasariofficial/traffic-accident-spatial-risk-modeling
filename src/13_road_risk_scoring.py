from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

INPUT_GPKG = (
    PROCESSED_DIR /
    "london_road_segment_network_features.gpkg"
)

MODEL_FILE = (
    MODEL_DIR /
    "best_road_risk_model.joblib"
)

THRESHOLD_FILE = (
    OUTPUT_DIR /
    "selected_threshold.txt"
)

OUTPUT_GPKG = (
    PROCESSED_DIR /
    "london_road_risk_scores.gpkg"
)

OUTPUT_CSV = (
    PROCESSED_DIR /
    "london_road_risk_scores.csv"
)

TOP_RISK_FILE = (
    OUTPUT_DIR /
    "top_risk_road_segments.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - ROAD RISK SCORING")
print("=" * 80)


# ============================================================
# 1. LOAD ROAD DATA
# ============================================================

if not INPUT_GPKG.exists():

    raise FileNotFoundError(
        f"Network feature dataset not found:\n"
        f"{INPUT_GPKG}"
    )


roads = gpd.read_file(
    INPUT_GPKG,
    layer="network_features"
)


print("\nRoad network feature dataset loaded successfully.")

print(
    f"\nRoad segments: "
    f"{len(roads):,}"
)

print(
    f"CRS: "
    f"{roads.crs}"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Best model not found:\n"
        f"{MODEL_FILE}"
    )


model = joblib.load(
    MODEL_FILE
)


print("\nBest road risk model loaded successfully.")


# ============================================================
# 3. LOAD SELECTED THRESHOLD
# ============================================================

if THRESHOLD_FILE.exists():

    with open(
        THRESHOLD_FILE,
        "r"
    ) as file:

        selected_threshold = float(
            file.read().strip()
        )

else:

    selected_threshold = 0.75


print(
    f"Selected operating threshold: "
    f"{selected_threshold:.2f}"
)


# ============================================================
# 4. MODEL FEATURES
# ============================================================

numeric_features = [

    "log_segment_length",
    "road_hierarchy",
    "is_major_road",
    "has_road_name",
    "is_oneway",
    "is_bridge",
    "is_tunnel",
    "maxspeed_numeric",
    "maxspeed_missing",
    "high_speed_osm",
    "mean_node_degree",
    "max_node_degree",
    "degree_difference",
    "touches_dead_end"
]


categorical_features = [
    "highway_clean"
]


all_features = (
    numeric_features +
    categorical_features
)


# ============================================================
# 5. FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)


missing_features = [

    feature

    for feature in all_features

    if feature not in roads.columns
]


if missing_features:

    raise KeyError(
        f"Missing model features: "
        f"{missing_features}"
    )


print(
    f"All {len(all_features)} model features available."
)


# ============================================================
# 6. PREDICT COLLISION PROBABILITY
# ============================================================

print("\n" + "=" * 80)
print("COLLISION RISK PREDICTION")
print("=" * 80)


X = (
    roads[
        all_features
    ]
    .copy()
)


roads[
    "collision_probability"
] = (
    model.predict_proba(
        X
    )[:, 1]
)


print(
    roads[
        "collision_probability"
    ]
    .describe()
    .round(4)
)


# ============================================================
# 7. RAW RISK SCORE
# ============================================================

roads[
    "risk_score_raw"
] = (
    roads[
        "collision_probability"
    ]
    * 100
)


print("\n" + "=" * 80)
print("RAW RISK SCORE")
print("=" * 80)


print(
    roads[
        "risk_score_raw"
    ]
    .describe()
    .round(2)
)


# ============================================================
# 8. PERCENTILE RISK SCORE
# ============================================================

# The model is trained with class balancing.
#
# Therefore raw probabilities are useful for ranking,
# but should not be interpreted as calibrated real-world
# collision probabilities.
#
# A percentile score provides an intuitive 0-100 relative
# risk ranking across London road segments.

roads[
    "risk_percentile"
] = (
    roads[
        "collision_probability"
    ]
    .rank(
        method="average",
        pct=True
    )
    * 100
)


print("\n" + "=" * 80)
print("RELATIVE RISK PERCENTILE")
print("=" * 80)


print(
    roads[
        "risk_percentile"
    ]
    .describe()
    .round(2)
)


# ============================================================
# 9. RISK CATEGORY
# ============================================================

# Relative risk classes:
#
# Low      = bottom 50%
# Moderate = 50-80%
# High     = 80-95%
# Critical = top 5%

roads[
    "risk_category"
] = pd.cut(

    roads[
        "risk_percentile"
    ],

    bins=[
        0,
        50,
        80,
        95,
        100
    ],

    labels=[
        "Low",
        "Moderate",
        "High",
        "Critical"
    ],

    include_lowest=True
)


print("\n" + "=" * 80)
print("RISK CATEGORY DISTRIBUTION")
print("=" * 80)


risk_counts = (
    roads[
        "risk_category"
    ]
    .value_counts()
    .reindex(
        [
            "Low",
            "Moderate",
            "High",
            "Critical"
        ]
    )
)


print(
    risk_counts
)


print("\nRisk category percentages:")


risk_percentages = (
    roads[
        "risk_category"
    ]
    .value_counts(
        normalize=True
    )
    .reindex(
        [
            "Low",
            "Moderate",
            "High",
            "Critical"
        ]
    )
    * 100
)


print(
    risk_percentages
    .round(2)
)


# ============================================================
# 10. OPERATING-THRESHOLD FLAG
# ============================================================

roads[
    "predicted_high_risk"
] = (
    roads[
        "collision_probability"
    ]
    >= selected_threshold
).astype(int)


print("\n" + "=" * 80)
print("OPERATING THRESHOLD CLASSIFICATION")
print("=" * 80)


print(
    f"Predicted high-risk segments: "
    f"{roads['predicted_high_risk'].sum():,}"
)

print(
    f"Predicted high-risk rate: "
    f"{roads['predicted_high_risk'].mean() * 100:.2f}%"
)


# ============================================================
# 11. OBSERVED COLLISION COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("OBSERVED COLLISIONS BY RISK CATEGORY")
print("=" * 80)


risk_summary = (
    roads
    .groupby(
        "risk_category",
        observed=False
    )
    .agg(

        Road_Segments=(
            "risk_category",
            "size"
        ),

        Observed_Collision_Segments=(
            "has_collision",
            "sum"
        ),

        Total_Collisions=(
            "collision_count",
            "sum"
        ),

        Severe_Collisions=(
            "severe_collision_count",
            "sum"
        ),

        Fatal_Collisions=(
            "fatal_collision_count",
            "sum"
        ),

        Mean_Raw_Risk_Score=(
            "risk_score_raw",
            "mean"
        )

    )
    .reset_index()
)


risk_summary[
    "Observed_Collision_Rate"
] = (
    risk_summary[
        "Observed_Collision_Segments"
    ]
    /
    risk_summary[
        "Road_Segments"
    ]
    * 100
)


risk_summary[
    "Collision_Share"
] = (
    risk_summary[
        "Total_Collisions"
    ]
    /
    roads[
        "collision_count"
    ]
    .sum()
    * 100
)


print(
    risk_summary
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. TOP 10% CAPTURE RATE
# ============================================================

print("\n" + "=" * 80)
print("TOP-RISK CAPTURE ANALYSIS")
print("=" * 80)


top_10_threshold = (
    roads[
        "risk_percentile"
    ]
    .quantile(
        0.90
    )
)


top_10 = (
    roads[
        roads[
            "risk_percentile"
        ] >= top_10_threshold
    ]
)


top_10_collision_share = (
    top_10[
        "collision_count"
    ]
    .sum()
    /
    roads[
        "collision_count"
    ]
    .sum()
    * 100
)


top_10_severe_share = (
    top_10[
        "severe_collision_count"
    ]
    .sum()
    /
    roads[
        "severe_collision_count"
    ]
    .sum()
    * 100
)


print(
    f"Top 10% road segments: "
    f"{len(top_10):,}"
)

print(
    f"Share of all collisions captured: "
    f"{top_10_collision_share:.2f}%"
)

print(
    f"Share of severe collisions captured: "
    f"{top_10_severe_share:.2f}%"
)


# ============================================================
# 13. TOP 5% CAPTURE RATE
# ============================================================

top_5 = (
    roads[
        roads[
            "risk_category"
        ] == "Critical"
    ]
)


top_5_collision_share = (
    top_5[
        "collision_count"
    ]
    .sum()
    /
    roads[
        "collision_count"
    ]
    .sum()
    * 100
)


top_5_severe_share = (
    top_5[
        "severe_collision_count"
    ]
    .sum()
    /
    roads[
        "severe_collision_count"
    ]
    .sum()
    * 100
)


print(
    f"\nCritical / top 5% segments: "
    f"{len(top_5):,}"
)

print(
    f"Share of all collisions captured: "
    f"{top_5_collision_share:.2f}%"
)

print(
    f"Share of severe collisions captured: "
    f"{top_5_severe_share:.2f}%"
)


# ============================================================
# 14. TOP RISK ROAD SEGMENTS
# ============================================================

print("\n" + "=" * 80)
print("TOP RISK ROAD SEGMENTS")
print("=" * 80)


display_columns = [

    column

    for column in [

        "road_segment_id",
        "name",
        "highway_clean",
        "segment_length_m",

        "collision_probability",
        "risk_score_raw",
        "risk_percentile",
        "risk_category",

        "collision_count",
        "severe_collision_count",
        "fatal_collision_count"

    ]

    if column in roads.columns
]


top_risk = (
    roads
    .sort_values(
        [
            "collision_probability",
            "collision_count"
        ],
        ascending=False
    )
    [
        display_columns
    ]
    .head(50)
)


print(
    top_risk
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


top_risk.to_csv(
    TOP_RISK_FILE,
    index=False
)


# ============================================================
# 15. RISK SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CREATE RISK SCORE DISTRIBUTION")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.hist(
    roads[
        "risk_score_raw"
    ],
    bins=50
)


ax.axvline(
    selected_threshold * 100,
    linestyle="--",
    linewidth=2,
    label=(
        f"Operating threshold "
        f"({selected_threshold * 100:.0f})"
    )
)


ax.set_title(
    "Distribution of Road Segment Risk Scores"
)

ax.set_xlabel(
    "Model Risk Score"
)

ax.set_ylabel(
    "Road Segment Frequency"
)

ax.legend()


plt.tight_layout()


risk_distribution_file = (
    FIGURE_DIR /
    "road_risk_score_distribution.png"
)


plt.savefig(
    risk_distribution_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Risk score distribution saved to:\n"
    f"{risk_distribution_file}"
)


# ============================================================
# 16. COLLISION RATE BY RISK CATEGORY
# ============================================================

print("\n" + "=" * 80)
print("CREATE RISK CATEGORY PERFORMANCE FIGURE")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


plot_summary = (
    risk_summary
    .set_index(
        "risk_category"
    )
)


ax.bar(
    plot_summary.index.astype(str),
    plot_summary[
        "Observed_Collision_Rate"
    ]
)


ax.set_title(
    "Observed Collision Rate by Predicted Risk Category"
)

ax.set_xlabel(
    "Predicted Risk Category"
)

ax.set_ylabel(
    "Road Segments with Collision (%)"
)


plt.tight_layout()


risk_category_file = (
    FIGURE_DIR /
    "collision_rate_by_risk_category.png"
)


plt.savefig(
    risk_category_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Risk category figure saved to:\n"
    f"{risk_category_file}"
)


# ============================================================
# 17. CREATE ROAD RISK MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE ROAD RISK MAP")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(12, 12)
)


# Plot all roads lightly first.

roads.plot(
    ax=ax,
    linewidth=0.15,
    alpha=0.20
)


# Overlay High and Critical roads.

high_risk = (
    roads[
        roads[
            "risk_category"
        ]
        .isin(
            [
                "High",
                "Critical"
            ]
        )
    ]
)


high_risk.plot(
    ax=ax,
    column="risk_percentile",
    linewidth=0.8,
    legend=True
)


ax.set_title(
    "London Predicted Road Collision Risk"
)

ax.set_axis_off()


plt.tight_layout()


risk_map_file = (
    MAP_DIR /
    "london_predicted_road_risk.png"
)


plt.savefig(
    risk_map_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Road risk map saved to:\n"
    f"{risk_map_file}"
)


# ============================================================
# 18. SAVE RISK SUMMARY
# ============================================================

risk_summary_file = (
    OUTPUT_DIR /
    "risk_category_summary.csv"
)


risk_summary.to_csv(
    risk_summary_file,
    index=False
)


# ============================================================
# 19. SAVE GEOPACKAGE
# ============================================================

print("\n" + "=" * 80)
print("SAVE ROAD RISK SCORES")
print("=" * 80)


roads.to_file(
    OUTPUT_GPKG,
    layer="road_risk_scores",
    driver="GPKG"
)


print(
    f"GeoPackage saved to:\n"
    f"{OUTPUT_GPKG}"
)


# ============================================================
# 20. SAVE CSV
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
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL ROAD RISK SCORING CHECK")
print("=" * 80)


print(
    f"Road segments scored: "
    f"{len(roads):,}"
)

print(
    f"Low risk: "
    f"{(roads['risk_category'] == 'Low').sum():,}"
)

print(
    f"Moderate risk: "
    f"{(roads['risk_category'] == 'Moderate').sum():,}"
)

print(
    f"High risk: "
    f"{(roads['risk_category'] == 'High').sum():,}"
)

print(
    f"Critical risk: "
    f"{(roads['risk_category'] == 'Critical').sum():,}"
)

print(
    f"Operating-threshold high-risk segments: "
    f"{roads['predicted_high_risk'].sum():,}"
)

print(
    f"Top 10% collision capture: "
    f"{top_10_collision_share:.2f}%"
)

print(
    f"Top 10% severe collision capture: "
    f"{top_10_severe_share:.2f}%"
)

print(
    f"Critical 5% collision capture: "
    f"{top_5_collision_share:.2f}%"
)

print(
    f"Critical 5% severe collision capture: "
    f"{top_5_severe_share:.2f}%"
)


print("\n" + "=" * 80)
print("ROAD RISK SCORING COMPLETED")
print("=" * 80)