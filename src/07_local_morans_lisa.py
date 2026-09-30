from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from esda.moran import Moran_Local


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

OUTPUT_LISA_GPKG = (
    PROCESSED_DATA_DIR /
    "london_lisa_results.gpkg"
)

OUTPUT_LISA_CSV = (
    PROCESSED_DATA_DIR /
    "london_lisa_results.csv"
)

OUTPUT_COLLISION_MAP = (
    OUTPUT_MAP_DIR /
    "lisa_collision_density.png"
)

OUTPUT_SEVERITY_MAP = (
    OUTPUT_MAP_DIR /
    "lisa_severity_density.png"
)

OUTPUT_COLLISION_SCATTER = (
    OUTPUT_FIGURE_DIR /
    "lisa_moran_scatter_collision_density.png"
)

OUTPUT_SEVERITY_SCATTER = (
    OUTPUT_FIGURE_DIR /
    "lisa_moran_scatter_severity_density.png"
)


# ============================================================
# SETTINGS
# ============================================================

PERMUTATIONS = 999

SIGNIFICANCE_LEVEL = 0.05

RANDOM_SEED = 42


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
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - LOCAL MORAN'S I / LISA")
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
            f"Required column not found: "
            f"{column}"
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
# 2. ANALYSIS DATA
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
# 3. SPATIAL WEIGHTS
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
# 5. STANDARDIZATION FUNCTION
# ============================================================

def standardize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = values.mean()

    std = values.std(
        ddof=0
    )

    if std == 0:

        raise ValueError(
            "Variable has zero variance."
        )

    return (
        values - mean
    ) / std


# ============================================================
# 6. COLLISION DENSITY VALUES
# ============================================================

print("\n" + "=" * 80)
print("COLLISION DENSITY STANDARDIZATION")
print("=" * 80)

collision_values = (
    analysis_gdf[
        "collision_density"
    ]
    .to_numpy()
)

collision_z = standardize(
    collision_values
)

analysis_gdf[
    "collision_z"
] = collision_z


print(
    f"Standardized mean: "
    f"{collision_z.mean():.6f}"
)

print(
    f"Standardized std: "
    f"{collision_z.std():.6f}"
)


# ============================================================
# 7. LOCAL MORAN - COLLISION DENSITY
# ============================================================

print("\n" + "=" * 80)
print("LOCAL MORAN'S I - COLLISION DENSITY")
print("=" * 80)

np.random.seed(
    RANDOM_SEED
)

collision_lisa = Moran_Local(
    collision_values,
    weights,
    permutations=PERMUTATIONS,
    seed=RANDOM_SEED
)


analysis_gdf[
    "collision_local_i"
] = collision_lisa.Is

analysis_gdf[
    "collision_p"
] = collision_lisa.p_sim

analysis_gdf[
    "collision_q"
] = collision_lisa.q


print(
    f"Local statistics calculated: "
    f"{len(collision_lisa.Is):,}"
)

print(
    f"Minimum Local Moran's I: "
    f"{collision_lisa.Is.min():.4f}"
)

print(
    f"Maximum Local Moran's I: "
    f"{collision_lisa.Is.max():.4f}"
)


# ============================================================
# 8. COLLISION SPATIAL LAG
# ============================================================

collision_lag = (
    weights.sparse @
    collision_z
)

analysis_gdf[
    "collision_lag_z"
] = collision_lag


# ============================================================
# 9. COLLISION LISA CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("COLLISION LISA CLASSIFICATION")
print("=" * 80)

collision_significant = (
    analysis_gdf[
        "collision_p"
    ] < SIGNIFICANCE_LEVEL
)

analysis_gdf[
    "collision_significant"
] = collision_significant.astype(int)


# PySAL Moran_Local quadrants:
#
# 1 = High-High
# 2 = Low-High
# 3 = Low-Low
# 4 = High-Low
#
# We classify only statistically significant cells.
# Non-significant cells are labelled NS.

quadrant_labels = {
    1: "High-High",
    2: "Low-High",
    3: "Low-Low",
    4: "High-Low"
}


analysis_gdf[
    "collision_lisa_cluster"
] = "NS"


for quadrant, label in quadrant_labels.items():

    mask = (
        collision_significant
        &
        (
            analysis_gdf[
                "collision_q"
            ] == quadrant
        )
    )

    analysis_gdf.loc[
        mask,
        "collision_lisa_cluster"
    ] = label


print(
    analysis_gdf[
        "collision_lisa_cluster"
    ]
    .value_counts()
)


# ============================================================
# 10. COLLISION CLUSTER COUNTS
# ============================================================

collision_cluster_counts = (
    analysis_gdf[
        "collision_lisa_cluster"
    ]
    .value_counts()
)

collision_hh = (
    collision_cluster_counts
    .get(
        "High-High",
        0
    )
)

collision_ll = (
    collision_cluster_counts
    .get(
        "Low-Low",
        0
    )
)

collision_hl = (
    collision_cluster_counts
    .get(
        "High-Low",
        0
    )
)

collision_lh = (
    collision_cluster_counts
    .get(
        "Low-High",
        0
    )
)

collision_ns = (
    collision_cluster_counts
    .get(
        "NS",
        0
    )
)


print(
    f"\nSignificant High-High cells: "
    f"{collision_hh:,}"
)

print(
    f"Significant Low-Low cells: "
    f"{collision_ll:,}"
)

print(
    f"Significant High-Low cells: "
    f"{collision_hl:,}"
)

print(
    f"Significant Low-High cells: "
    f"{collision_lh:,}"
)

print(
    f"Non-significant cells: "
    f"{collision_ns:,}"
)


# ============================================================
# 11. SEVERITY DENSITY VALUES
# ============================================================

print("\n" + "=" * 80)
print("SEVERITY DENSITY STANDARDIZATION")
print("=" * 80)

severity_values = (
    analysis_gdf[
        "severity_density"
    ]
    .to_numpy()
)

severity_z = standardize(
    severity_values
)

analysis_gdf[
    "severity_z"
] = severity_z


print(
    f"Standardized mean: "
    f"{severity_z.mean():.6f}"
)

print(
    f"Standardized std: "
    f"{severity_z.std():.6f}"
)


# ============================================================
# 12. LOCAL MORAN - SEVERITY DENSITY
# ============================================================

print("\n" + "=" * 80)
print("LOCAL MORAN'S I - SEVERITY-WEIGHTED DENSITY")
print("=" * 80)

np.random.seed(
    RANDOM_SEED
)

severity_lisa = Moran_Local(
    severity_values,
    weights,
    permutations=PERMUTATIONS,
    seed=RANDOM_SEED
)


analysis_gdf[
    "severity_local_i"
] = severity_lisa.Is

analysis_gdf[
    "severity_p"
] = severity_lisa.p_sim

analysis_gdf[
    "severity_q"
] = severity_lisa.q


print(
    f"Local statistics calculated: "
    f"{len(severity_lisa.Is):,}"
)

print(
    f"Minimum Local Moran's I: "
    f"{severity_lisa.Is.min():.4f}"
)

print(
    f"Maximum Local Moran's I: "
    f"{severity_lisa.Is.max():.4f}"
)


# ============================================================
# 13. SEVERITY SPATIAL LAG
# ============================================================

severity_lag = (
    weights.sparse @
    severity_z
)

analysis_gdf[
    "severity_lag_z"
] = severity_lag


# ============================================================
# 14. SEVERITY LISA CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("SEVERITY LISA CLASSIFICATION")
print("=" * 80)

severity_significant = (
    analysis_gdf[
        "severity_p"
    ] < SIGNIFICANCE_LEVEL
)

analysis_gdf[
    "severity_significant"
] = severity_significant.astype(int)


analysis_gdf[
    "severity_lisa_cluster"
] = "NS"


for quadrant, label in quadrant_labels.items():

    mask = (
        severity_significant
        &
        (
            analysis_gdf[
                "severity_q"
            ] == quadrant
        )
    )

    analysis_gdf.loc[
        mask,
        "severity_lisa_cluster"
    ] = label


print(
    analysis_gdf[
        "severity_lisa_cluster"
    ]
    .value_counts()
)


# ============================================================
# 15. SEVERITY CLUSTER COUNTS
# ============================================================

severity_cluster_counts = (
    analysis_gdf[
        "severity_lisa_cluster"
    ]
    .value_counts()
)

severity_hh = (
    severity_cluster_counts
    .get(
        "High-High",
        0
    )
)

severity_ll = (
    severity_cluster_counts
    .get(
        "Low-Low",
        0
    )
)

severity_hl = (
    severity_cluster_counts
    .get(
        "High-Low",
        0
    )
)

severity_lh = (
    severity_cluster_counts
    .get(
        "Low-High",
        0
    )
)

severity_ns = (
    severity_cluster_counts
    .get(
        "NS",
        0
    )
)


print(
    f"\nSignificant High-High cells: "
    f"{severity_hh:,}"
)

print(
    f"Significant Low-Low cells: "
    f"{severity_ll:,}"
)

print(
    f"Significant High-Low cells: "
    f"{severity_hl:,}"
)

print(
    f"Significant Low-High cells: "
    f"{severity_lh:,}"
)

print(
    f"Non-significant cells: "
    f"{severity_ns:,}"
)


# ============================================================
# 16. OVERLAP BETWEEN COLLISION AND SEVERITY HH CLUSTERS
# ============================================================

print("\n" + "=" * 80)
print("HIGH-HIGH CLUSTER OVERLAP")
print("=" * 80)

collision_hh_mask = (
    analysis_gdf[
        "collision_lisa_cluster"
    ] == "High-High"
)

severity_hh_mask = (
    analysis_gdf[
        "severity_lisa_cluster"
    ] == "High-High"
)


analysis_gdf[
    "both_high_high"
] = (
    collision_hh_mask
    &
    severity_hh_mask
).astype(int)


both_hh = (
    analysis_gdf[
        "both_high_high"
    ]
    .sum()
)


print(
    f"Collision-density HH cells: "
    f"{collision_hh:,}"
)

print(
    f"Severity-density HH cells: "
    f"{severity_hh:,}"
)

print(
    f"Cells High-High in both analyses: "
    f"{both_hh:,}"
)


if collision_hh > 0:

    collision_overlap_rate = (
        both_hh /
        collision_hh *
        100
    )

else:

    collision_overlap_rate = np.nan


if severity_hh > 0:

    severity_overlap_rate = (
        both_hh /
        severity_hh *
        100
    )

else:

    severity_overlap_rate = np.nan


print(
    f"Collision HH overlap rate: "
    f"{collision_overlap_rate:.2f}%"
)

print(
    f"Severity HH overlap rate: "
    f"{severity_overlap_rate:.2f}%"
)


# ============================================================
# 17. PRIORITY CLUSTER FLAG
# ============================================================

# A priority spatial cluster is a cell that is:
#
# 1. Significant High-High for collision density
# AND
# 2. Significant High-High for severity density
#
# This remains a statistical prioritisation indicator,
# not a causal risk estimate.

analysis_gdf[
    "priority_cluster"
] = (
    analysis_gdf[
        "both_high_high"
    ]
)


print(
    f"\nPriority cluster cells: "
    f"{analysis_gdf['priority_cluster'].sum():,}"
)


# ============================================================
# 18. TOP PRIORITY CELLS
# ============================================================

print("\n" + "=" * 80)
print("TOP PRIORITY LISA CELLS")
print("=" * 80)

priority_cells = (
    analysis_gdf[
        analysis_gdf[
            "priority_cluster"
        ] == 1
    ]
    .copy()
)


if not priority_cells.empty:

    top_priority = (
        priority_cells
        .sort_values(
            [
                "severity_density",
                "collision_density"
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
                "collision_local_i",
                "collision_p",
                "severity_local_i",
                "severity_p"
            ]
        ]
        .head(20)
    )

    print(
        top_priority
        .to_string(
            index=False
        )
    )

else:

    print(
        "No overlapping High-High "
        "priority cells detected."
    )


# ============================================================
# 19. COLLISION LISA MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE COLLISION LISA MAP")
print("=" * 80)

# Explicit colors are used here because LISA cluster maps
# conventionally require categorical differentiation.

collision_colors = {
    "High-High": "#d7191c",
    "Low-Low": "#2c7bb6",
    "High-Low": "#fdae61",
    "Low-High": "#abd9e9",
    "NS": "#d9d9d9"
}


fig, ax = plt.subplots(
    figsize=(11, 11)
)


for cluster in [
    "NS",
    "Low-Low",
    "Low-High",
    "High-Low",
    "High-High"
]:

    subset = (
        analysis_gdf[
            analysis_gdf[
                "collision_lisa_cluster"
            ] == cluster
        ]
    )

    if subset.empty:
        continue

    subset.plot(
        ax=ax,
        color=collision_colors[
            cluster
        ],
        linewidth=0,
        label=cluster
    )


ax.set_title(
    "Local Moran's I (LISA) — London Collision Density, 2025",
    fontsize=15
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)

ax.legend(
    title="LISA Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_COLLISION_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Collision LISA map saved to:\n"
    f"{OUTPUT_COLLISION_MAP}"
)


# ============================================================
# 20. SEVERITY LISA MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE SEVERITY LISA MAP")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(11, 11)
)


for cluster in [
    "NS",
    "Low-Low",
    "Low-High",
    "High-Low",
    "High-High"
]:

    subset = (
        analysis_gdf[
            analysis_gdf[
                "severity_lisa_cluster"
            ] == cluster
        ]
    )

    if subset.empty:
        continue

    subset.plot(
        ax=ax,
        color=collision_colors[
            cluster
        ],
        linewidth=0,
        label=cluster
    )


ax.set_title(
    "Local Moran's I (LISA) — London Severity-Weighted Density, 2025",
    fontsize=15
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)

ax.legend(
    title="LISA Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_SEVERITY_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Severity LISA map saved to:\n"
    f"{OUTPUT_SEVERITY_MAP}"
)


# ============================================================
# 21. COLLISION MORAN SCATTERPLOT
# ============================================================

print("\n" + "=" * 80)
print("CREATE COLLISION MORAN SCATTERPLOT")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

ax.scatter(
    analysis_gdf[
        "collision_z"
    ],
    analysis_gdf[
        "collision_lag_z"
    ],
    s=8,
    alpha=0.35
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)


# Regression line

slope, intercept = np.polyfit(
    analysis_gdf[
        "collision_z"
    ],
    analysis_gdf[
        "collision_lag_z"
    ],
    1
)

x_line = np.linspace(
    analysis_gdf[
        "collision_z"
    ].min(),
    analysis_gdf[
        "collision_z"
    ].max(),
    100
)

y_line = (
    slope * x_line +
    intercept
)

ax.plot(
    x_line,
    y_line,
    linewidth=2
)


ax.set_title(
    "Moran Scatterplot — Collision Density"
)

ax.set_xlabel(
    "Standardized Collision Density"
)

ax.set_ylabel(
    "Spatial Lag of Standardized Collision Density"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_COLLISION_SCATTER,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Collision Moran scatterplot saved to:\n"
    f"{OUTPUT_COLLISION_SCATTER}"
)


# ============================================================
# 22. SEVERITY MORAN SCATTERPLOT
# ============================================================

print("\n" + "=" * 80)
print("CREATE SEVERITY MORAN SCATTERPLOT")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(9, 8)
)

ax.scatter(
    analysis_gdf[
        "severity_z"
    ],
    analysis_gdf[
        "severity_lag_z"
    ],
    s=8,
    alpha=0.35
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)


slope, intercept = np.polyfit(
    analysis_gdf[
        "severity_z"
    ],
    analysis_gdf[
        "severity_lag_z"
    ],
    1
)

x_line = np.linspace(
    analysis_gdf[
        "severity_z"
    ].min(),
    analysis_gdf[
        "severity_z"
    ].max(),
    100
)

y_line = (
    slope * x_line +
    intercept
)

ax.plot(
    x_line,
    y_line,
    linewidth=2
)


ax.set_title(
    "Moran Scatterplot — Severity-Weighted Density"
)

ax.set_xlabel(
    "Standardized Severity-Weighted Density"
)

ax.set_ylabel(
    "Spatial Lag of Standardized Severity Density"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_SEVERITY_SCATTER,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Severity Moran scatterplot saved to:\n"
    f"{OUTPUT_SEVERITY_SCATTER}"
)


# ============================================================
# 23. SAVE GEOPACKAGE
# ============================================================

print("\n" + "=" * 80)
print("SAVE LISA RESULTS")
print("=" * 80)

analysis_gdf.to_file(
    OUTPUT_LISA_GPKG,
    layer="lisa_results",
    driver="GPKG"
)

print(
    f"LISA GeoPackage saved to:\n"
    f"{OUTPUT_LISA_GPKG}"
)


# ============================================================
# 24. SAVE CSV
# ============================================================

lisa_csv = (
    analysis_gdf
    .drop(
        columns="geometry"
    )
    .copy()
)

lisa_csv.to_csv(
    OUTPUT_LISA_CSV,
    index=False
)

print(
    f"\nLISA CSV saved to:\n"
    f"{OUTPUT_LISA_CSV}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL LISA CHECK")
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
    f"Significance level: "
    f"{SIGNIFICANCE_LEVEL}"
)

print(
    f"Collision High-High cells: "
    f"{collision_hh:,}"
)

print(
    f"Severity High-High cells: "
    f"{severity_hh:,}"
)

print(
    f"Overlapping High-High cells: "
    f"{both_hh:,}"
)

print(
    f"Priority cluster cells: "
    f"{analysis_gdf['priority_cluster'].sum():,}"
)

print(
    f"Collision significant cells: "
    f"{analysis_gdf['collision_significant'].sum():,}"
)

print(
    f"Severity significant cells: "
    f"{analysis_gdf['severity_significant'].sum():,}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("LOCAL MORAN'S I / LISA ANALYSIS COMPLETED")
print("=" * 80)