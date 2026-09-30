from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

INPUT_COLLISIONS = (
    PROCESSED_DATA_DIR /
    "london_collisions_road_matched.gpkg"
)

OUTPUT_GRID = (
    PROCESSED_DATA_DIR /
    "london_accident_density_grid.gpkg"
)

OUTPUT_COLLISION_MAP = (
    OUTPUT_MAP_DIR /
    "london_collision_density.png"
)

OUTPUT_SEVERITY_MAP = (
    OUTPUT_MAP_DIR /
    "london_severity_weighted_density.png"
)

OUTPUT_HOTSPOT_MAP = (
    OUTPUT_MAP_DIR /
    "london_high_density_areas.png"
)

OUTPUT_DISTRIBUTION = (
    OUTPUT_FIGURE_DIR /
    "collision_density_distribution.png"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_ROAD_DISTANCE = 50

# 250 x 250 metre analysis cells
GRID_SIZE = 250

# Gaussian smoothing expressed in number of grid cells.
# sigma=2 with a 250 m grid corresponds to approximately
# 500 m smoothing scale.
SMOOTHING_SIGMA = 2


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
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - ACCIDENT HOTSPOTS")
print("=" * 80)

if not INPUT_COLLISIONS.exists():

    raise FileNotFoundError(
        f"Road-matched collision dataset not found:\n"
        f"{INPUT_COLLISIONS}"
    )


gdf = gpd.read_file(
    INPUT_COLLISIONS,
    layer="collision_road_matches"
)

print("\nRoad-matched collision dataset loaded successfully.")

print(f"\nOriginal collisions: {len(gdf):,}")
print(f"CRS: {gdf.crs}")


# ============================================================
# 1. ROAD MATCH QUALITY FILTER
# ============================================================

print("\n" + "=" * 80)
print("ROAD MATCH QUALITY FILTER")
print("=" * 80)

before = len(gdf)

analysis_gdf = (
    gdf[
        gdf["distance_to_road_m"] <= MAX_ROAD_DISTANCE
    ]
    .copy()
)

removed = before - len(analysis_gdf)

print(
    f"Maximum accepted collision-to-road distance: "
    f"{MAX_ROAD_DISTANCE} m"
)

print(
    f"Collisions retained: "
    f"{len(analysis_gdf):,}"
)

print(
    f"Collisions excluded: "
    f"{removed:,}"
)

print(
    f"Retention rate: "
    f"{len(analysis_gdf) / before * 100:.2f}%"
)


# ============================================================
# 2. SEVERITY WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("SEVERITY WEIGHTING")
print("=" * 80)

# Analytical weighting used only for the weighted-density
# surface. It is not an official DfT risk score.
#
# Slight  = 1
# Serious = 3
# Fatal   = 5

severity_weight_map = {
    "Slight": 1,
    "Serious": 3,
    "Fatal": 5
}

analysis_gdf["severity_weight"] = (
    analysis_gdf["severity_label"]
    .map(severity_weight_map)
)


if analysis_gdf["severity_weight"].isna().any():

    raise ValueError(
        "Some severity labels could not be assigned "
        "a severity weight."
    )


print(
    analysis_gdf[
        [
            "severity_label",
            "severity_weight"
        ]
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 3. SPATIAL EXTENT
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS EXTENT")
print("=" * 80)

minx, miny, maxx, maxy = (
    analysis_gdf.total_bounds
)

print(f"Minimum easting: {minx:,.2f}")
print(f"Maximum easting: {maxx:,.2f}")

print(f"Minimum northing: {miny:,.2f}")
print(f"Maximum northing: {maxy:,.2f}")


# ============================================================
# 4. CREATE REGULAR GRID
# ============================================================

print("\n" + "=" * 80)
print("REGULAR GRID CREATION")
print("=" * 80)

x_edges = np.arange(
    np.floor(minx / GRID_SIZE) * GRID_SIZE,
    np.ceil(maxx / GRID_SIZE) * GRID_SIZE + GRID_SIZE,
    GRID_SIZE
)

y_edges = np.arange(
    np.floor(miny / GRID_SIZE) * GRID_SIZE,
    np.ceil(maxy / GRID_SIZE) * GRID_SIZE + GRID_SIZE,
    GRID_SIZE
)


print(
    f"Grid cell size: "
    f"{GRID_SIZE} x {GRID_SIZE} metres"
)

print(
    f"Grid columns: "
    f"{len(x_edges) - 1:,}"
)

print(
    f"Grid rows: "
    f"{len(y_edges) - 1:,}"
)

print(
    f"Total rectangular extent cells: "
    f"{(len(x_edges) - 1) * (len(y_edges) - 1):,}"
)


# ============================================================
# 5. COLLISION COUNT GRID
# ============================================================

print("\n" + "=" * 80)
print("COLLISION COUNT SURFACE")
print("=" * 80)

x = analysis_gdf.geometry.x.to_numpy()
y = analysis_gdf.geometry.y.to_numpy()


collision_histogram, _, _ = np.histogram2d(
    x,
    y,
    bins=[
        x_edges,
        y_edges
    ]
)

# histogram2d returns x dimension first.
# Transpose for standard raster orientation.

collision_histogram = (
    collision_histogram.T
)


print(
    f"Collisions represented in grid: "
    f"{int(collision_histogram.sum()):,}"
)

print(
    f"Maximum raw collisions in one cell: "
    f"{int(collision_histogram.max()):,}"
)


# ============================================================
# 6. SEVERITY-WEIGHTED GRID
# ============================================================

print("\n" + "=" * 80)
print("SEVERITY-WEIGHTED SURFACE")
print("=" * 80)

severity_histogram, _, _ = np.histogram2d(
    x,
    y,
    bins=[
        x_edges,
        y_edges
    ],
    weights=(
        analysis_gdf[
            "severity_weight"
        ]
        .to_numpy()
    )
)

severity_histogram = (
    severity_histogram.T
)


print(
    f"Total severity weight: "
    f"{severity_histogram.sum():,.0f}"
)

print(
    f"Maximum raw severity weight in one cell: "
    f"{severity_histogram.max():,.0f}"
)


# ============================================================
# 7. GAUSSIAN DENSITY SURFACES
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL SMOOTHING")
print("=" * 80)

collision_density = gaussian_filter(
    collision_histogram,
    sigma=SMOOTHING_SIGMA
)

severity_density = gaussian_filter(
    severity_histogram,
    sigma=SMOOTHING_SIGMA
)


print(
    f"Smoothing sigma: "
    f"{SMOOTHING_SIGMA} grid cells"
)

print(
    f"Approximate smoothing scale: "
    f"{SMOOTHING_SIGMA * GRID_SIZE:,.0f} m"
)

print(
    f"Maximum smoothed collision density: "
    f"{collision_density.max():.4f}"
)

print(
    f"Maximum smoothed severity density: "
    f"{severity_density.max():.4f}"
)


# ============================================================
# 8. DENSITY PERCENTILES
# ============================================================

print("\n" + "=" * 80)
print("DENSITY THRESHOLDS")
print("=" * 80)

positive_collision_density = (
    collision_density[
        collision_density > 0
    ]
)

positive_severity_density = (
    severity_density[
        severity_density > 0
    ]
)


collision_p90 = np.percentile(
    positive_collision_density,
    90
)

collision_p95 = np.percentile(
    positive_collision_density,
    95
)

collision_p99 = np.percentile(
    positive_collision_density,
    99
)


severity_p90 = np.percentile(
    positive_severity_density,
    90
)

severity_p95 = np.percentile(
    positive_severity_density,
    95
)

severity_p99 = np.percentile(
    positive_severity_density,
    99
)


print("Collision density:")

print(f"90th percentile: {collision_p90:.4f}")
print(f"95th percentile: {collision_p95:.4f}")
print(f"99th percentile: {collision_p99:.4f}")


print("\nSeverity-weighted density:")

print(f"90th percentile: {severity_p90:.4f}")
print(f"95th percentile: {severity_p95:.4f}")
print(f"99th percentile: {severity_p99:.4f}")


# ============================================================
# 9. CREATE CELL-LEVEL DATASET
# ============================================================

print("\n" + "=" * 80)
print("GRID DATASET CREATION")
print("=" * 80)

records = []

for row_index in range(
    collision_density.shape[0]
):

    for column_index in range(
        collision_density.shape[1]
    ):

        cell_minx = (
            x_edges[column_index]
        )

        cell_maxx = (
            x_edges[column_index + 1]
        )

        cell_miny = (
            y_edges[row_index]
        )

        cell_maxy = (
            y_edges[row_index + 1]
        )

        records.append(
            {
                "grid_row": row_index,
                "grid_col": column_index,

                "cell_minx": cell_minx,
                "cell_maxx": cell_maxx,
                "cell_miny": cell_miny,
                "cell_maxy": cell_maxy,

                "collision_count": int(
                    collision_histogram[
                        row_index,
                        column_index
                    ]
                ),

                "severity_weight_sum": float(
                    severity_histogram[
                        row_index,
                        column_index
                    ]
                ),

                "collision_density": float(
                    collision_density[
                        row_index,
                        column_index
                    ]
                ),

                "severity_density": float(
                    severity_density[
                        row_index,
                        column_index
                    ]
                )
            }
        )


grid_df = pd.DataFrame(
    records
)


# ============================================================
# 10. HOTSPOT CANDIDATE FLAGS
# ============================================================

# IMPORTANT:
#
# These are density-based candidate areas.
# They are NOT yet statistically significant hotspots.
#
# Statistical significance will be tested later using
# Local Moran's I and Getis-Ord Gi*.

grid_df[
    "high_collision_density"
] = (
    grid_df[
        "collision_density"
    ] >= collision_p95
).astype(int)


grid_df[
    "high_severity_density"
] = (
    grid_df[
        "severity_density"
    ] >= severity_p95
).astype(int)


grid_df[
    "combined_density_candidate"
] = (
    (
        grid_df[
            "high_collision_density"
        ] == 1
    )
    &
    (
        grid_df[
            "high_severity_density"
        ] == 1
    )
).astype(int)


print(
    f"High collision-density cells: "
    f"{grid_df['high_collision_density'].sum():,}"
)

print(
    f"High severity-density cells: "
    f"{grid_df['high_severity_density'].sum():,}"
)

print(
    f"Combined high-density candidate cells: "
    f"{grid_df['combined_density_candidate'].sum():,}"
)


# ============================================================
# 11. CREATE GRID GEOMETRY
# ============================================================

print("\n" + "=" * 80)
print("GRID GEOMETRY")
print("=" * 80)

from shapely.geometry import box


grid_geometry = [
    box(
        row.cell_minx,
        row.cell_miny,
        row.cell_maxx,
        row.cell_maxy
    )

    for row in grid_df.itertuples()
]


grid_gdf = gpd.GeoDataFrame(
    grid_df,
    geometry=grid_geometry,
    crs="EPSG:27700"
)


# Keep cells with some raw collision activity OR
# meaningful smoothed density.

grid_gdf = (
    grid_gdf[
        (
            grid_gdf["collision_count"] > 0
        )
        |
        (
            grid_gdf["collision_density"] > 0.001
        )
    ]
    .copy()
)


print(
    f"Grid cells retained for spatial analysis: "
    f"{len(grid_gdf):,}"
)


# ============================================================
# 12. SAVE GRID
# ============================================================

print("\n" + "=" * 80)
print("SAVE DENSITY GRID")
print("=" * 80)

grid_gdf.to_file(
    OUTPUT_GRID,
    layer="accident_density_grid",
    driver="GPKG"
)

print(
    f"Density grid saved to:\n"
    f"{OUTPUT_GRID}"
)


# ============================================================
# 13. COLLISION DENSITY MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE COLLISION DENSITY MAP")
print("=" * 80)

extent = [
    x_edges[0],
    x_edges[-1],
    y_edges[0],
    y_edges[-1]
]


fig, ax = plt.subplots(
    figsize=(11, 11)
)

image = ax.imshow(
    collision_density,
    origin="lower",
    extent=extent,
    interpolation="bilinear"
)

ax.set_title(
    "London Traffic Collision Density — 2025",
    fontsize=16
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)

colorbar = fig.colorbar(
    image,
    ax=ax,
    shrink=0.75
)

colorbar.set_label(
    "Smoothed Collision Density"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_COLLISION_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Collision density map saved to:\n"
    f"{OUTPUT_COLLISION_MAP}"
)


# ============================================================
# 14. SEVERITY-WEIGHTED DENSITY MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE SEVERITY-WEIGHTED MAP")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(11, 11)
)

image = ax.imshow(
    severity_density,
    origin="lower",
    extent=extent,
    interpolation="bilinear"
)

ax.set_title(
    "London Severity-Weighted Collision Density — 2025",
    fontsize=16
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)

colorbar = fig.colorbar(
    image,
    ax=ax,
    shrink=0.75
)

colorbar.set_label(
    "Severity-Weighted Spatial Density"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_SEVERITY_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Severity-weighted density map saved to:\n"
    f"{OUTPUT_SEVERITY_MAP}"
)


# ============================================================
# 15. HIGH-DENSITY CANDIDATE MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE HIGH-DENSITY CANDIDATE MAP")
print("=" * 80)

candidate_gdf = (
    grid_gdf[
        grid_gdf[
            "combined_density_candidate"
        ] == 1
    ]
    .copy()
)


fig, ax = plt.subplots(
    figsize=(11, 11)
)

grid_gdf.plot(
    ax=ax,
    column="collision_density",
    linewidth=0
)

if not candidate_gdf.empty:

    candidate_gdf.boundary.plot(
        ax=ax,
        linewidth=0.8
    )


ax.set_title(
    "London High Collision-Density Candidate Areas — 2025",
    fontsize=16
)

ax.set_xlabel(
    "British National Grid Easting (m)"
)

ax.set_ylabel(
    "British National Grid Northing (m)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_HOTSPOT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Candidate-area map saved to:\n"
    f"{OUTPUT_HOTSPOT_MAP}"
)


# ============================================================
# 16. DENSITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CREATE DENSITY DISTRIBUTION")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    positive_collision_density,
    bins=50
)

ax.axvline(
    collision_p95,
    linestyle="--",
    linewidth=2,
    label="95th percentile"
)

ax.set_title(
    "Distribution of Smoothed Collision Density"
)

ax.set_xlabel(
    "Collision Density"
)

ax.set_ylabel(
    "Grid Cell Frequency"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DISTRIBUTION,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Density distribution saved to:\n"
    f"{OUTPUT_DISTRIBUTION}"
)


# ============================================================
# 17. TOP DENSITY CELLS
# ============================================================

print("\n" + "=" * 80)
print("TOP COLLISION-DENSITY CELLS")
print("=" * 80)

top_collision_cells = (
    grid_gdf
    .sort_values(
        "collision_density",
        ascending=False
    )
    [
        [
            "grid_row",
            "grid_col",
            "collision_count",
            "severity_weight_sum",
            "collision_density",
            "severity_density"
        ]
    ]
    .head(15)
)

print(
    top_collision_cells
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("TOP SEVERITY-DENSITY CELLS")
print("=" * 80)

top_severity_cells = (
    grid_gdf
    .sort_values(
        "severity_density",
        ascending=False
    )
    [
        [
            "grid_row",
            "grid_col",
            "collision_count",
            "severity_weight_sum",
            "collision_density",
            "severity_density"
        ]
    ]
    .head(15)
)

print(
    top_severity_cells
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL HOTSPOT CHECK")
print("=" * 80)

print(
    f"Original London collisions: "
    f"{len(gdf):,}"
)

print(
    f"Collisions used in spatial density analysis: "
    f"{len(analysis_gdf):,}"
)

print(
    f"Excluded road-match outliers: "
    f"{removed:,}"
)

print(
    f"Grid cell size: "
    f"{GRID_SIZE} m"
)

print(
    f"Retained grid cells: "
    f"{len(grid_gdf):,}"
)

print(
    f"Combined high-density candidate cells: "
    f"{grid_gdf['combined_density_candidate'].sum():,}"
)

print(
    f"Maximum collision density: "
    f"{grid_gdf['collision_density'].max():.4f}"
)

print(
    f"Maximum severity density: "
    f"{grid_gdf['severity_density'].max():.4f}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("ACCIDENT HOTSPOT ANALYSIS COMPLETED")
print("=" * 80)