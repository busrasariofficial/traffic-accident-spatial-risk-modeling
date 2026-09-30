from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from libpysal.weights import Queen
from esda.getisord import G_Local


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

OUTPUT_MAP_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "maps"
)

OUTPUT_FIGURE_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "figures"
)


INPUT_GRID = (
    PROCESSED_DATA_DIR /
    "london_accident_density_grid.gpkg"
)

OUTPUT_GI_GPKG = (
    PROCESSED_DATA_DIR /
    "london_getis_ord_gi_star_results.gpkg"
)

OUTPUT_GI_CSV = (
    PROCESSED_DATA_DIR /
    "london_getis_ord_gi_star_results.csv"
)

OUTPUT_COLLISION_MAP = (
    OUTPUT_MAP_DIR /
    "getis_ord_collision_hotspots.png"
)

OUTPUT_SEVERITY_MAP = (
    OUTPUT_MAP_DIR /
    "getis_ord_severity_hotspots.png"
)

OUTPUT_COLLISION_Z = (
    OUTPUT_FIGURE_DIR /
    "getis_ord_collision_z_distribution.png"
)

OUTPUT_SEVERITY_Z = (
    OUTPUT_FIGURE_DIR /
    "getis_ord_severity_z_distribution.png"
)


# ============================================================
# SETTINGS
# ============================================================

PERMUTATIONS = 999
RANDOM_SEED = 42

SIGNIFICANCE_90 = 0.10
SIGNIFICANCE_95 = 0.05
SIGNIFICANCE_99 = 0.01


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_MAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD GRID
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - GETIS-ORD GI*")
print("=" * 80)

if not INPUT_GRID.exists():

    raise FileNotFoundError(
        f"Density grid not found:\n"
        f"{INPUT_GRID}"
    )


gdf = gpd.read_file(
    INPUT_GRID,
    layer="accident_density_grid"
)


print("\nDensity grid loaded successfully.")

print(
    f"\nGrid cells: "
    f"{len(gdf):,}"
)

print(
    f"CRS: "
    f"{gdf.crs}"
)


# ============================================================
# 1. VARIABLE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VARIABLE VALIDATION")
print("=" * 80)

required_columns = [
    "collision_density",
    "severity_density"
]


for column in required_columns:

    if column not in gdf.columns:

        raise KeyError(
            f"Required column not found: {column}"
        )

    missing = (
        gdf[column]
        .isna()
        .sum()
    )

    print(
        f"{column} missing values: "
        f"{missing:,}"
    )


# ============================================================
# 2. ANALYSIS DATA PREPARATION
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS DATA PREPARATION")
print("=" * 80)

before = len(gdf)

analysis_gdf = (
    gdf[
        gdf[
            required_columns
        ]
        .notna()
        .all(axis=1)
    ]
    .copy()
    .reset_index(drop=True)
)

removed = (
    before -
    len(analysis_gdf)
)


print(
    f"Cells retained: "
    f"{len(analysis_gdf):,}"
)

print(
    f"Cells removed: "
    f"{removed:,}"
)


# ============================================================
# 3. SPATIAL WEIGHTS MATRIX
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL WEIGHTS MATRIX")
print("=" * 80)

weights = Queen.from_dataframe(
    analysis_gdf,
    use_index=False
)


print(
    f"Spatial units: "
    f"{weights.n:,}"
)

print(
    f"Total neighbour links: "
    f"{weights.s0:,.0f}"
)

print(
    f"Mean neighbours per cell: "
    f"{weights.mean_neighbors:.2f}"
)

print(
    f"Minimum neighbours: "
    f"{weights.min_neighbors}"
)

print(
    f"Maximum neighbours: "
    f"{weights.max_neighbors}"
)

print(
    f"Islands: "
    f"{len(weights.islands):,}"
)


# ============================================================
# 4. ROW STANDARDIZATION
# ============================================================

weights.transform = "R"

print(
    "\nSpatial weights row-standardized."
)


# ============================================================
# 5. GI* CLASSIFICATION FUNCTION
# ============================================================

def classify_gi_star(z_score, p_value):

    """
    Classify Getis-Ord Gi* results.

    Positive significant z-score:
        Hot Spot

    Negative significant z-score:
        Cold Spot

    Confidence levels:
        99% -> p < 0.01
        95% -> p < 0.05
        90% -> p < 0.10

    Non-significant:
        NS
    """

    if pd.isna(z_score) or pd.isna(p_value):

        return "NS"


    if p_value < SIGNIFICANCE_99:

        if z_score > 0:

            return "Hot Spot 99%"

        elif z_score < 0:

            return "Cold Spot 99%"


    elif p_value < SIGNIFICANCE_95:

        if z_score > 0:

            return "Hot Spot 95%"

        elif z_score < 0:

            return "Cold Spot 95%"


    elif p_value < SIGNIFICANCE_90:

        if z_score > 0:

            return "Hot Spot 90%"

        elif z_score < 0:

            return "Cold Spot 90%"


    return "NS"


# ============================================================
# 6. COLLISION DENSITY GI*
# ============================================================

print("\n" + "=" * 80)
print("GETIS-ORD GI* - COLLISION DENSITY")
print("=" * 80)

collision_values = (
    analysis_gdf[
        "collision_density"
    ]
    .to_numpy()
)


np.random.seed(
    RANDOM_SEED
)


collision_gi = G_Local(
    collision_values,
    weights,
    transform="R",
    permutations=PERMUTATIONS,
    star=True,
    seed=RANDOM_SEED
)


analysis_gdf[
    "collision_gi"
] = collision_gi.Gs

analysis_gdf[
    "collision_gi_z"
] = collision_gi.Zs

analysis_gdf[
    "collision_gi_p"
] = collision_gi.p_sim


print(
    f"Local Gi* statistics calculated: "
    f"{len(collision_gi.Gs):,}"
)

print(
    f"Minimum Gi* z-score: "
    f"{np.nanmin(collision_gi.Zs):.4f}"
)

print(
    f"Maximum Gi* z-score: "
    f"{np.nanmax(collision_gi.Zs):.4f}"
)

print(
    f"Minimum permutation p-value: "
    f"{np.nanmin(collision_gi.p_sim):.4f}"
)


# ============================================================
# 7. COLLISION GI* CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("COLLISION GI* CLASSIFICATION")
print("=" * 80)

analysis_gdf[
    "collision_gi_class"
] = [
    classify_gi_star(
        z,
        p
    )
    for z, p in zip(
        analysis_gdf[
            "collision_gi_z"
        ],
        analysis_gdf[
            "collision_gi_p"
        ]
    )
]


print(
    analysis_gdf[
        "collision_gi_class"
    ]
    .value_counts()
)


# ============================================================
# 8. COLLISION HOT SPOT FLAGS
# ============================================================

analysis_gdf[
    "collision_hotspot_95"
] = (
    (
        analysis_gdf[
            "collision_gi_z"
        ] > 0
    )
    &
    (
        analysis_gdf[
            "collision_gi_p"
        ] < 0.05
    )
).astype(int)


analysis_gdf[
    "collision_hotspot_99"
] = (
    (
        analysis_gdf[
            "collision_gi_z"
        ] > 0
    )
    &
    (
        analysis_gdf[
            "collision_gi_p"
        ] < 0.01
    )
).astype(int)


collision_hot_95 = (
    analysis_gdf[
        "collision_hotspot_95"
    ]
    .sum()
)

collision_hot_99 = (
    analysis_gdf[
        "collision_hotspot_99"
    ]
    .sum()
)


print(
    f"\nCollision hot spots "
    f"(p < 0.05): "
    f"{collision_hot_95:,}"
)

print(
    f"Collision hot spots "
    f"(p < 0.01): "
    f"{collision_hot_99:,}"
)


# ============================================================
# 9. SEVERITY-WEIGHTED GI*
# ============================================================

print("\n" + "=" * 80)
print("GETIS-ORD GI* - SEVERITY-WEIGHTED DENSITY")
print("=" * 80)

severity_values = (
    analysis_gdf[
        "severity_density"
    ]
    .to_numpy()
)


np.random.seed(
    RANDOM_SEED
)


severity_gi = G_Local(
    severity_values,
    weights,
    transform="R",
    permutations=PERMUTATIONS,
    star=True,
    seed=RANDOM_SEED
)


analysis_gdf[
    "severity_gi"
] = severity_gi.Gs

analysis_gdf[
    "severity_gi_z"
] = severity_gi.Zs

analysis_gdf[
    "severity_gi_p"
] = severity_gi.p_sim


print(
    f"Local Gi* statistics calculated: "
    f"{len(severity_gi.Gs):,}"
)

print(
    f"Minimum Gi* z-score: "
    f"{np.nanmin(severity_gi.Zs):.4f}"
)

print(
    f"Maximum Gi* z-score: "
    f"{np.nanmax(severity_gi.Zs):.4f}"
)

print(
    f"Minimum permutation p-value: "
    f"{np.nanmin(severity_gi.p_sim):.4f}"
)


# ============================================================
# 10. SEVERITY GI* CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("SEVERITY GI* CLASSIFICATION")
print("=" * 80)

analysis_gdf[
    "severity_gi_class"
] = [
    classify_gi_star(
        z,
        p
    )
    for z, p in zip(
        analysis_gdf[
            "severity_gi_z"
        ],
        analysis_gdf[
            "severity_gi_p"
        ]
    )
]


print(
    analysis_gdf[
        "severity_gi_class"
    ]
    .value_counts()
)


# ============================================================
# 11. SEVERITY HOT SPOT FLAGS
# ============================================================

analysis_gdf[
    "severity_hotspot_95"
] = (
    (
        analysis_gdf[
            "severity_gi_z"
        ] > 0
    )
    &
    (
        analysis_gdf[
            "severity_gi_p"
        ] < 0.05
    )
).astype(int)


analysis_gdf[
    "severity_hotspot_99"
] = (
    (
        analysis_gdf[
            "severity_gi_z"
        ] > 0
    )
    &
    (
        analysis_gdf[
            "severity_gi_p"
        ] < 0.01
    )
).astype(int)


severity_hot_95 = (
    analysis_gdf[
        "severity_hotspot_95"
    ]
    .sum()
)

severity_hot_99 = (
    analysis_gdf[
        "severity_hotspot_99"
    ]
    .sum()
)


print(
    f"\nSeverity hot spots "
    f"(p < 0.05): "
    f"{severity_hot_95:,}"
)

print(
    f"Severity hot spots "
    f"(p < 0.01): "
    f"{severity_hot_99:,}"
)


# ============================================================
# 12. COLLISION + SEVERITY HOT SPOT OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("HOT SPOT OVERLAP")
print("=" * 80)

analysis_gdf[
    "both_hotspot_95"
] = (
    (
        analysis_gdf[
            "collision_hotspot_95"
        ] == 1
    )
    &
    (
        analysis_gdf[
            "severity_hotspot_95"
        ] == 1
    )
).astype(int)


analysis_gdf[
    "both_hotspot_99"
] = (
    (
        analysis_gdf[
            "collision_hotspot_99"
        ] == 1
    )
    &
    (
        analysis_gdf[
            "severity_hotspot_99"
        ] == 1
    )
).astype(int)


both_hot_95 = (
    analysis_gdf[
        "both_hotspot_95"
    ]
    .sum()
)

both_hot_99 = (
    analysis_gdf[
        "both_hotspot_99"
    ]
    .sum()
)


print(
    f"Collision hot spots 95%: "
    f"{collision_hot_95:,}"
)

print(
    f"Severity hot spots 95%: "
    f"{severity_hot_95:,}"
)

print(
    f"Overlapping hot spots 95%: "
    f"{both_hot_95:,}"
)


print(
    f"\nCollision hot spots 99%: "
    f"{collision_hot_99:,}"
)

print(
    f"Severity hot spots 99%: "
    f"{severity_hot_99:,}"
)

print(
    f"Overlapping hot spots 99%: "
    f"{both_hot_99:,}"
)


# ============================================================
# 13. COMBINE WITH LISA PRIORITY
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL PRIORITY INDEX")
print("=" * 80)

LISA_FILE = (
    PROCESSED_DATA_DIR /
    "london_lisa_results.gpkg"
)


if LISA_FILE.exists():

    lisa_gdf = gpd.read_file(
        LISA_FILE,
        layer="lisa_results"
    )


    lisa_priority = (
        lisa_gdf[
            [
                "grid_row",
                "grid_col",
                "priority_cluster"
            ]
        ]
        .copy()
    )


    analysis_gdf = analysis_gdf.merge(
        lisa_priority,
        on=[
            "grid_row",
            "grid_col"
        ],
        how="left"
    )


    analysis_gdf[
        "priority_cluster"
    ] = (
        analysis_gdf[
            "priority_cluster"
        ]
        .fillna(0)
        .astype(int)
    )


else:

    print(
        "WARNING: LISA result file not found."
    )

    analysis_gdf[
        "priority_cluster"
    ] = 0


# ============================================================
# 14. FINAL CONSENSUS HOT SPOT
# ============================================================

# Strongest spatial-priority class:
#
# - Significant High-High in both LISA analyses
# - Significant Gi* hot spot in both density analyses
# - p < 0.01 for both Gi* tests
#
# This provides agreement across two different local
# spatial-statistical approaches.

analysis_gdf[
    "consensus_hotspot"
] = (
    (
        analysis_gdf[
            "priority_cluster"
        ] == 1
    )
    &
    (
        analysis_gdf[
            "both_hotspot_99"
        ] == 1
    )
).astype(int)


consensus_count = (
    analysis_gdf[
        "consensus_hotspot"
    ]
    .sum()
)


print(
    f"LISA priority cells: "
    f"{analysis_gdf['priority_cluster'].sum():,}"
)

print(
    f"Gi* overlapping 99% hot spots: "
    f"{both_hot_99:,}"
)

print(
    f"Final consensus hot spots: "
    f"{consensus_count:,}"
)


# ============================================================
# 15. TOP CONSENSUS HOT SPOTS
# ============================================================

print("\n" + "=" * 80)
print("TOP CONSENSUS HOT SPOTS")
print("=" * 80)

consensus_gdf = (
    analysis_gdf[
        analysis_gdf[
            "consensus_hotspot"
        ] == 1
    ]
    .copy()
)


if not consensus_gdf.empty:

    top_consensus = (
        consensus_gdf
        .sort_values(
            [
                "severity_gi_z",
                "collision_gi_z",
                "severity_density"
            ],
            ascending=False
        )
        [
            [
                "grid_row",
                "grid_col",
                "collision_count",
                "severity_weight_sum",
                "collision_density",
                "severity_density",
                "collision_gi_z",
                "collision_gi_p",
                "severity_gi_z",
                "severity_gi_p"
            ]
        ]
        .head(20)
    )


    print(
        top_consensus
        .to_string(
            index=False
        )
    )


else:

    print(
        "No final consensus hot spots detected."
    )


# ============================================================
# 16. MAP COLOR SETTINGS
# ============================================================

gi_colors = {

    "Hot Spot 99%": "#b2182b",
    "Hot Spot 95%": "#ef8a62",
    "Hot Spot 90%": "#fddbc7",

    "Cold Spot 99%": "#2166ac",
    "Cold Spot 95%": "#67a9cf",
    "Cold Spot 90%": "#d1e5f0",

    "NS": "#d9d9d9"
}


plot_order = [

    "NS",

    "Cold Spot 90%",
    "Cold Spot 95%",
    "Cold Spot 99%",

    "Hot Spot 90%",
    "Hot Spot 95%",
    "Hot Spot 99%"
]


# ============================================================
# 17. COLLISION GI* MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE COLLISION GI* MAP")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(11, 11)
)


for category in plot_order:

    subset = (
        analysis_gdf[
            analysis_gdf[
                "collision_gi_class"
            ] == category
        ]
    )


    if subset.empty:

        continue


    subset.plot(
        ax=ax,
        color=gi_colors[
            category
        ],
        linewidth=0
    )


legend_elements = [

    Patch(
        facecolor=gi_colors[category],
        edgecolor="none",
        label=category
    )

    for category in plot_order

    if category in (
        analysis_gdf[
            "collision_gi_class"
        ]
        .unique()
    )
]


ax.legend(
    handles=legend_elements,
    title="Getis-Ord Gi*",
    loc="lower left"
)


ax.set_title(
    "Getis-Ord Gi* Hot Spot Analysis — Collision Density",
    fontsize=15
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_COLLISION_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Collision Gi* map saved to:\n"
    f"{OUTPUT_COLLISION_MAP}"
)


# ============================================================
# 18. SEVERITY GI* MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE SEVERITY GI* MAP")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(11, 11)
)


for category in plot_order:

    subset = (
        analysis_gdf[
            analysis_gdf[
                "severity_gi_class"
            ] == category
        ]
    )


    if subset.empty:

        continue


    subset.plot(
        ax=ax,
        color=gi_colors[
            category
        ],
        linewidth=0
    )


legend_elements = [

    Patch(
        facecolor=gi_colors[category],
        edgecolor="none",
        label=category
    )

    for category in plot_order

    if category in (
        analysis_gdf[
            "severity_gi_class"
        ]
        .unique()
    )
]


ax.legend(
    handles=legend_elements,
    title="Getis-Ord Gi*",
    loc="lower left"
)


ax.set_title(
    "Getis-Ord Gi* Hot Spot Analysis — Severity-Weighted Density",
    fontsize=15
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_SEVERITY_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Severity Gi* map saved to:\n"
    f"{OUTPUT_SEVERITY_MAP}"
)


# ============================================================
# 19. COLLISION Z-SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CREATE COLLISION GI* Z-SCORE DISTRIBUTION")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.hist(
    analysis_gdf[
        "collision_gi_z"
    ]
    .dropna(),
    bins=50
)


ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)


ax.set_title(
    "Distribution of Local Gi* Z-Scores — Collision Density"
)

ax.set_xlabel(
    "Gi* Z-Score"
)

ax.set_ylabel(
    "Grid Cell Frequency"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_COLLISION_Z,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Collision z-score distribution saved to:\n"
    f"{OUTPUT_COLLISION_Z}"
)


# ============================================================
# 20. SEVERITY Z-SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CREATE SEVERITY GI* Z-SCORE DISTRIBUTION")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.hist(
    analysis_gdf[
        "severity_gi_z"
    ]
    .dropna(),
    bins=50
)


ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)


ax.set_title(
    "Distribution of Local Gi* Z-Scores — Severity-Weighted Density"
)

ax.set_xlabel(
    "Gi* Z-Score"
)

ax.set_ylabel(
    "Grid Cell Frequency"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_SEVERITY_Z,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Severity z-score distribution saved to:\n"
    f"{OUTPUT_SEVERITY_Z}"
)


# ============================================================
# 21. SAVE GEOPACKAGE
# ============================================================

print("\n" + "=" * 80)
print("SAVE GETIS-ORD RESULTS")
print("=" * 80)

analysis_gdf.to_file(
    OUTPUT_GI_GPKG,
    layer="getis_ord_gi_star",
    driver="GPKG"
)


print(
    f"Gi* GeoPackage saved to:\n"
    f"{OUTPUT_GI_GPKG}"
)


# ============================================================
# 22. SAVE CSV
# ============================================================

gi_csv = (
    analysis_gdf
    .drop(
        columns="geometry"
    )
    .copy()
)


gi_csv.to_csv(
    OUTPUT_GI_CSV,
    index=False
)


print(
    f"\nGi* CSV saved to:\n"
    f"{OUTPUT_GI_CSV}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL GETIS-ORD GI* CHECK")
print("=" * 80)


print(
    f"Spatial units analysed: "
    f"{len(analysis_gdf):,}"
)

print(
    f"Permutations: "
    f"{PERMUTATIONS:,}"
)

print(
    f"Collision hot spots p < 0.05: "
    f"{collision_hot_95:,}"
)

print(
    f"Collision hot spots p < 0.01: "
    f"{collision_hot_99:,}"
)

print(
    f"Severity hot spots p < 0.05: "
    f"{severity_hot_95:,}"
)

print(
    f"Severity hot spots p < 0.01: "
    f"{severity_hot_99:,}"
)

print(
    f"Overlapping hot spots p < 0.01: "
    f"{both_hot_99:,}"
)

print(
    f"LISA priority cells: "
    f"{analysis_gdf['priority_cluster'].sum():,}"
)

print(
    f"Final consensus hot spots: "
    f"{consensus_count:,}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("GETIS-ORD GI* ANALYSIS COMPLETED")
print("=" * 80)