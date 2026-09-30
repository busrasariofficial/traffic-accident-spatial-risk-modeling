from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from esda.moran import Moran


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FIGURE_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "figures"
)

INPUT_GRID = (
    PROCESSED_DATA_DIR /
    "london_accident_density_grid.gpkg"
)

OUTPUT_COLLISION_FIGURE = (
    OUTPUT_FIGURE_DIR /
    "global_morans_i_collision_density.png"
)

OUTPUT_SEVERITY_FIGURE = (
    OUTPUT_FIGURE_DIR /
    "global_morans_i_severity_density.png"
)


# ============================================================
# SETTINGS
# ============================================================

PERMUTATIONS = 999

RANDOM_SEED = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD GRID
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - GLOBAL MORAN'S I")
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
# 1. REQUIRED COLUMN CHECK
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
# 2. REMOVE INVALID VALUES
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS DATA PREPARATION")
print("=" * 80)

before = len(gdf)

analysis_gdf = (
    gdf[
        gdf[
            [
                "collision_density",
                "severity_density"
            ]
        ]
        .notna()
        .all(axis=1)
    ]
    .copy()
    .reset_index(drop=True)
)

removed = before - len(analysis_gdf)

print(
    f"Cells retained: "
    f"{len(analysis_gdf):,}"
)

print(
    f"Cells removed: "
    f"{removed:,}"
)


# ============================================================
# 3. CREATE SPATIAL WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL WEIGHTS MATRIX")
print("=" * 80)

# Queen contiguity:
#
# Two grid cells are considered neighbours if they
# share either:
#
# - an edge
# OR
# - a corner
#
# This is suitable for our regular 250 m grid.

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


# ============================================================
# 4. ISLAND CHECK
# ============================================================

print("\n" + "=" * 80)
print("ISLAND CHECK")
print("=" * 80)

islands = weights.islands

print(
    f"Cells without neighbours: "
    f"{len(islands):,}"
)

if len(islands) > 0:

    print(
        "\nWARNING: Isolated cells detected."
    )

    print(
        "Island indices:"
    )

    print(
        islands[:20]
    )


# ============================================================
# 5. ROW-STANDARDIZE WEIGHTS
# ============================================================

weights.transform = "R"

print(
    "\nSpatial weights transformed using "
    "row standardization."
)


# ============================================================
# 6. VARIABLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("VARIABLE SUMMARY")
print("=" * 80)

print("\nCollision density:")

print(
    analysis_gdf[
        "collision_density"
    ]
    .describe()
    .round(4)
)


print("\nSeverity density:")

print(
    analysis_gdf[
        "severity_density"
    ]
    .describe()
    .round(4)
)


# ============================================================
# 7. GLOBAL MORAN'S I - COLLISION DENSITY
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MORAN'S I - COLLISION DENSITY")
print("=" * 80)

np.random.seed(
    RANDOM_SEED
)

collision_values = (
    analysis_gdf[
        "collision_density"
    ]
    .to_numpy()
)

collision_moran = Moran(
    collision_values,
    weights,
    permutations=PERMUTATIONS
)


print(
    f"Moran's I: "
    f"{collision_moran.I:.6f}"
)

print(
    f"Expected I: "
    f"{collision_moran.EI:.6f}"
)

print(
    f"Permutation p-value: "
    f"{collision_moran.p_sim:.6f}"
)

print(
    f"Simulation mean: "
    f"{collision_moran.EI_sim:.6f}"
)

print(
    f"Simulation standard deviation: "
    f"{collision_moran.seI_sim:.6f}"
)

print(
    f"Simulation z-score: "
    f"{collision_moran.z_sim:.4f}"
)


# ============================================================
# 8. COLLISION DENSITY INTERPRETATION
# ============================================================

print("\nInterpretation:")

if collision_moran.p_sim < 0.05:

    if collision_moran.I > 0:

        print(
            "Collision density shows statistically "
            "significant positive spatial autocorrelation."
        )

        print(
            "Similar collision-density values tend to "
            "cluster spatially."
        )

    elif collision_moran.I < 0:

        print(
            "Collision density shows statistically "
            "significant negative spatial autocorrelation."
        )

        print(
            "Neighbouring cells tend to have "
            "dissimilar collision-density values."
        )

    else:

        print(
            "The observed spatial pattern is statistically "
            "significant but Moran's I is approximately zero."
        )

else:

    print(
        "No statistically significant global spatial "
        "autocorrelation was detected."
    )


# ============================================================
# 9. GLOBAL MORAN'S I - SEVERITY DENSITY
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MORAN'S I - SEVERITY-WEIGHTED DENSITY")
print("=" * 80)

np.random.seed(
    RANDOM_SEED
)

severity_values = (
    analysis_gdf[
        "severity_density"
    ]
    .to_numpy()
)

severity_moran = Moran(
    severity_values,
    weights,
    permutations=PERMUTATIONS
)


print(
    f"Moran's I: "
    f"{severity_moran.I:.6f}"
)

print(
    f"Expected I: "
    f"{severity_moran.EI:.6f}"
)

print(
    f"Permutation p-value: "
    f"{severity_moran.p_sim:.6f}"
)

print(
    f"Simulation mean: "
    f"{severity_moran.EI_sim:.6f}"
)

print(
    f"Simulation standard deviation: "
    f"{severity_moran.seI_sim:.6f}"
)

print(
    f"Simulation z-score: "
    f"{severity_moran.z_sim:.4f}"
)


# ============================================================
# 10. SEVERITY DENSITY INTERPRETATION
# ============================================================

print("\nInterpretation:")

if severity_moran.p_sim < 0.05:

    if severity_moran.I > 0:

        print(
            "Severity-weighted density shows statistically "
            "significant positive spatial autocorrelation."
        )

        print(
            "Areas with similar severity-weighted density "
            "tend to occur near one another."
        )

    elif severity_moran.I < 0:

        print(
            "Severity-weighted density shows statistically "
            "significant negative spatial autocorrelation."
        )

    else:

        print(
            "The observed pattern is statistically "
            "significant but Moran's I is approximately zero."
        )

else:

    print(
        "No statistically significant global spatial "
        "autocorrelation was detected."
    )


# ============================================================
# 11. COMPARE GLOBAL MORAN RESULTS
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MORAN COMPARISON")
print("=" * 80)

print(
    f"Collision density Moran's I: "
    f"{collision_moran.I:.6f}"
)

print(
    f"Severity density Moran's I: "
    f"{severity_moran.I:.6f}"
)

difference = (
    severity_moran.I
    - collision_moran.I
)

print(
    f"Difference "
    f"(severity - collision): "
    f"{difference:.6f}"
)


# ============================================================
# 12. PERMUTATION DISTRIBUTION - COLLISION DENSITY
# ============================================================

print("\n" + "=" * 80)
print("CREATE COLLISION MORAN PERMUTATION FIGURE")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    collision_moran.sim,
    bins=40
)

ax.axvline(
    collision_moran.I,
    linestyle="--",
    linewidth=2,
    label=(
        f"Observed Moran's I = "
        f"{collision_moran.I:.3f}"
    )
)

ax.axvline(
    collision_moran.EI_sim,
    linestyle=":",
    linewidth=2,
    label=(
        f"Permutation mean = "
        f"{collision_moran.EI_sim:.3f}"
    )
)

ax.set_title(
    "Global Moran's I Permutation Test — Collision Density"
)

ax.set_xlabel(
    "Moran's I under Spatial Randomization"
)

ax.set_ylabel(
    "Frequency"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_COLLISION_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Collision Moran figure saved to:\n"
    f"{OUTPUT_COLLISION_FIGURE}"
)


# ============================================================
# 13. PERMUTATION DISTRIBUTION - SEVERITY DENSITY
# ============================================================

print("\n" + "=" * 80)
print("CREATE SEVERITY MORAN PERMUTATION FIGURE")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    severity_moran.sim,
    bins=40
)

ax.axvline(
    severity_moran.I,
    linestyle="--",
    linewidth=2,
    label=(
        f"Observed Moran's I = "
        f"{severity_moran.I:.3f}"
    )
)

ax.axvline(
    severity_moran.EI_sim,
    linestyle=":",
    linewidth=2,
    label=(
        f"Permutation mean = "
        f"{severity_moran.EI_sim:.3f}"
    )
)

ax.set_title(
    "Global Moran's I Permutation Test — Severity-Weighted Density"
)

ax.set_xlabel(
    "Moran's I under Spatial Randomization"
)

ax.set_ylabel(
    "Frequency"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_SEVERITY_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Severity Moran figure saved to:\n"
    f"{OUTPUT_SEVERITY_FIGURE}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL GLOBAL MORAN'S I CHECK")
print("=" * 80)

print(
    f"Spatial units analysed: "
    f"{len(analysis_gdf):,}"
)

print(
    f"Queen contiguity islands: "
    f"{len(islands):,}"
)

print(
    f"Permutations: "
    f"{PERMUTATIONS:,}"
)

print(
    f"Collision density Moran's I: "
    f"{collision_moran.I:.6f}"
)

print(
    f"Collision density p-value: "
    f"{collision_moran.p_sim:.6f}"
)

print(
    f"Severity density Moran's I: "
    f"{severity_moran.I:.6f}"
)

print(
    f"Severity density p-value: "
    f"{severity_moran.p_sim:.6f}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MORAN'S I ANALYSIS COMPLETED")
print("=" * 80)