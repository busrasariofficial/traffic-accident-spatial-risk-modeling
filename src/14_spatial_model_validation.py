from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"

RISK_FILE = (
    PROCESSED_DIR /
    "london_road_risk_scores.gpkg"
)

DENSITY_FILE = (
    PROCESSED_DIR /
    "london_accident_density_grid.gpkg"
)

VALIDATION_GPKG = (
    PROCESSED_DIR /
    "london_road_risk_spatial_validation.gpkg"
)

VALIDATION_CSV = (
    OUTPUT_DIR /
    "spatial_validation_summary.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - SPATIAL MODEL VALIDATION")
print("=" * 80)


# ============================================================
# 1. LOAD ROAD RISK SCORES
# ============================================================

if not RISK_FILE.exists():

    raise FileNotFoundError(
        f"Road risk file not found:\n"
        f"{RISK_FILE}"
    )


roads = gpd.read_file(
    RISK_FILE,
    layer="road_risk_scores"
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
# 2. LOAD ACCIDENT DENSITY GRID
# ============================================================

if not DENSITY_FILE.exists():

    raise FileNotFoundError(
        f"Density grid not found:\n"
        f"{DENSITY_FILE}"
    )


grid = gpd.read_file(
    DENSITY_FILE
)


print("\nAccident density grid loaded successfully.")

print(
    f"\nGrid cells: "
    f"{len(grid):,}"
)

print(
    f"Grid CRS: "
    f"{grid.crs}"
)


# ============================================================
# 3. CRS VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("CRS VALIDATION")
print("=" * 80)


if roads.crs != grid.crs:

    print(
        "CRS mismatch detected. "
        "Transforming grid..."
    )

    grid = grid.to_crs(
        roads.crs
    )


print(
    f"Road CRS: "
    f"{roads.crs}"
)

print(
    f"Grid CRS: "
    f"{grid.crs}"
)


# ============================================================
# 4. GRID COLUMN VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("GRID VARIABLE VALIDATION")
print("=" * 80)


required_grid_columns = [

    "collision_density",
    "severity_density"
]


for column in required_grid_columns:

    if column not in grid.columns:

        raise KeyError(
            f"Required grid column missing: "
            f"{column}"
        )

    print(
        f"[OK] {column}"
    )


# ============================================================
# 5. CREATE ROAD MIDPOINTS
# ============================================================

print("\n" + "=" * 80)
print("ROAD MIDPOINT CREATION")
print("=" * 80)


road_points = roads.copy()


road_points[
    "original_geometry"
] = road_points.geometry


road_points[
    "geometry"
] = (
    road_points
    .geometry
    .interpolate(
        0.5,
        normalized=True
    )
)


print(
    f"Road midpoint points created: "
    f"{len(road_points):,}"
)


# ============================================================
# 6. SPATIAL JOIN WITH DENSITY GRID
# ============================================================

print("\n" + "=" * 80)
print("ROAD-TO-HOTSPOT SPATIAL JOIN")
print("=" * 80)


grid_columns = [

    "collision_density",
    "severity_density",
    "high_collision_density",
    "high_severity_density",
    "combined_high_density",
    "geometry"
]


grid_columns = [

    column

    for column in grid_columns

    if column in grid.columns
]


joined = gpd.sjoin(

    road_points,

    grid[
        grid_columns
    ],

    how="left",

    predicate="within"
)


joined = (
    joined
    .drop_duplicates(
        subset="road_segment_id"
    )
    .copy()
)


print(
    f"Road segments after spatial join: "
    f"{len(joined):,}"
)


print(
    f"Segments matched to density grid: "
    f"{joined['collision_density'].notna().sum():,}"
)


# ============================================================
# 7. RESTORE ROAD GEOMETRY
# ============================================================

joined[
    "geometry"
] = joined[
    "original_geometry"
]


joined = joined.drop(
    columns=[
        "original_geometry",
        "index_right"
    ],
    errors="ignore"
)


joined = gpd.GeoDataFrame(
    joined,
    geometry="geometry",
    crs=roads.crs
)


# ============================================================
# 8. MISSING DENSITY HANDLING
# ============================================================

print("\n" + "=" * 80)
print("DENSITY MATCH QUALITY")
print("=" * 80)


missing_density = (
    joined[
        "collision_density"
    ]
    .isna()
    .sum()
)


print(
    f"Segments without density match: "
    f"{missing_density:,}"
)


for column in [

    "collision_density",
    "severity_density",
    "high_collision_density",
    "high_severity_density",
    "combined_high_density"

]:

    if column in joined.columns:

        joined[column] = (
            joined[column]
            .fillna(0)
        )


# ============================================================
# 9. SPEARMAN SPATIAL ASSOCIATION
# ============================================================

print("\n" + "=" * 80)
print("RISK SCORE VS SPATIAL DENSITY")
print("=" * 80)


collision_corr, collision_p = (
    spearmanr(
        joined[
            "risk_percentile"
        ],
        joined[
            "collision_density"
        ]
    )
)


severity_corr, severity_p = (
    spearmanr(
        joined[
            "risk_percentile"
        ],
        joined[
            "severity_density"
        ]
    )
)


print(
    f"Risk percentile vs collision density:"
)

print(
    f"Spearman rho: "
    f"{collision_corr:.4f}"
)

print(
    f"p-value: "
    f"{collision_p:.6g}"
)


print(
    "\nRisk percentile vs severity density:"
)

print(
    f"Spearman rho: "
    f"{severity_corr:.4f}"
)

print(
    f"p-value: "
    f"{severity_p:.6g}"
)


# ============================================================
# 10. HOTSPOT FLAGS
# ============================================================

print("\n" + "=" * 80)
print("HOTSPOT VALIDATION FLAGS")
print("=" * 80)


if "combined_high_density" in joined.columns:

    joined[
        "observed_hotspot"
    ] = (
        joined[
            "combined_high_density"
        ] == 1
    ).astype(int)

else:

    collision_threshold = (
        grid[
            "collision_density"
        ]
        .quantile(
            0.95
        )
    )


    severity_threshold = (
        grid[
            "severity_density"
        ]
        .quantile(
            0.95
        )
    )


    joined[
        "observed_hotspot"
    ] = (
        (
            joined[
                "collision_density"
            ] >= collision_threshold
        )
        &
        (
            joined[
                "severity_density"
            ] >= severity_threshold
        )
    ).astype(int)


joined[
    "model_high_risk"
] = (
    joined[
        "risk_category"
    ]
    .isin(
        [
            "High",
            "Critical"
        ]
    )
).astype(int)


joined[
    "model_critical"
] = (
    joined[
        "risk_category"
    ] == "Critical"
).astype(int)


print(
    f"Observed hotspot road segments: "
    f"{joined['observed_hotspot'].sum():,}"
)

print(
    f"Model High/Critical segments: "
    f"{joined['model_high_risk'].sum():,}"
)

print(
    f"Model Critical segments: "
    f"{joined['model_critical'].sum():,}"
)


# ============================================================
# 11. HOTSPOT OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("MODEL / HOTSPOT OVERLAP")
print("=" * 80)


high_risk_hotspot_overlap = (

    (
        (
            joined[
                "model_high_risk"
            ] == 1
        )
        &
        (
            joined[
                "observed_hotspot"
            ] == 1
        )
    )
    .sum()
)


critical_hotspot_overlap = (

    (
        (
            joined[
                "model_critical"
            ] == 1
        )
        &
        (
            joined[
                "observed_hotspot"
            ] == 1
        )
    )
    .sum()
)


high_risk_hotspot_rate = (

    high_risk_hotspot_overlap
    /
    joined[
        "model_high_risk"
    ]
    .sum()
    * 100

    if joined[
        "model_high_risk"
    ]
    .sum() > 0

    else 0
)


critical_hotspot_rate = (

    critical_hotspot_overlap
    /
    joined[
        "model_critical"
    ]
    .sum()
    * 100

    if joined[
        "model_critical"
    ]
    .sum() > 0

    else 0
)


print(
    f"High/Critical roads inside observed hotspots: "
    f"{high_risk_hotspot_overlap:,}"
)

print(
    f"High/Critical hotspot overlap rate: "
    f"{high_risk_hotspot_rate:.2f}%"
)


print(
    f"\nCritical roads inside observed hotspots: "
    f"{critical_hotspot_overlap:,}"
)

print(
    f"Critical hotspot overlap rate: "
    f"{critical_hotspot_rate:.2f}%"
)


# ============================================================
# 12. HOTSPOT CAPTURE RATE
# ============================================================

total_hotspots = (
    joined[
        "observed_hotspot"
    ]
    .sum()
)


hotspots_captured_high = (
    high_risk_hotspot_overlap
)


hotspot_capture_rate_high = (

    hotspots_captured_high
    /
    total_hotspots
    * 100

    if total_hotspots > 0

    else 0
)


hotspot_capture_rate_critical = (

    critical_hotspot_overlap
    /
    total_hotspots
    * 100

    if total_hotspots > 0

    else 0
)


print("\n" + "=" * 80)
print("HOTSPOT CAPTURE RATE")
print("=" * 80)


print(
    f"Observed hotspot segments: "
    f"{total_hotspots:,}"
)

print(
    f"Captured by High/Critical prediction: "
    f"{hotspots_captured_high:,}"
)

print(
    f"High/Critical hotspot capture rate: "
    f"{hotspot_capture_rate_high:.2f}%"
)

print(
    f"Critical-only hotspot capture rate: "
    f"{hotspot_capture_rate_critical:.2f}%"
)


# ============================================================
# 13. COLLISION CAPTURE BY MODEL RISK
# ============================================================

print("\n" + "=" * 80)
print("COLLISION CAPTURE BY MODEL RISK")
print("=" * 80)


total_collisions = (
    joined[
        "collision_count"
    ]
    .sum()
)


total_severe = (
    joined[
        "severe_collision_count"
    ]
    .sum()
)


high_risk_roads = (
    joined[
        joined[
            "model_high_risk"
        ] == 1
    ]
)


critical_roads = (
    joined[
        joined[
            "model_critical"
        ] == 1
    ]
)


high_collision_capture = (

    high_risk_roads[
        "collision_count"
    ]
    .sum()
    /
    total_collisions
    * 100
)


high_severe_capture = (

    high_risk_roads[
        "severe_collision_count"
    ]
    .sum()
    /
    total_severe
    * 100
)


critical_collision_capture = (

    critical_roads[
        "collision_count"
    ]
    .sum()
    /
    total_collisions
    * 100
)


critical_severe_capture = (

    critical_roads[
        "severe_collision_count"
    ]
    .sum()
    /
    total_severe
    * 100
)


print(
    f"High/Critical collision capture: "
    f"{high_collision_capture:.2f}%"
)

print(
    f"High/Critical severe collision capture: "
    f"{high_severe_capture:.2f}%"
)

print(
    f"Critical collision capture: "
    f"{critical_collision_capture:.2f}%"
)

print(
    f"Critical severe collision capture: "
    f"{critical_severe_capture:.2f}%"
)


# ============================================================
# 14. RISK CATEGORY VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("RISK CATEGORY SPATIAL VALIDATION")
print("=" * 80)


category_validation = (

    joined
    .groupby(
        "risk_category",
        observed=False
    )
    .agg(

        Road_Segments=(
            "road_segment_id",
            "count"
        ),

        Mean_Collision_Density=(
            "collision_density",
            "mean"
        ),

        Mean_Severity_Density=(
            "severity_density",
            "mean"
        ),

        Hotspot_Roads=(
            "observed_hotspot",
            "sum"
        ),

        Observed_Collisions=(
            "collision_count",
            "sum"
        ),

        Severe_Collisions=(
            "severe_collision_count",
            "sum"
        )

    )
    .reset_index()
)


category_validation[
    "Hotspot_Rate"
] = (

    category_validation[
        "Hotspot_Roads"
    ]
    /
    category_validation[
        "Road_Segments"
    ]
    * 100
)


print(
    category_validation
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. CREATE SPATIAL VALIDATION MAP
# ============================================================

print("\n" + "=" * 80)
print("CREATE SPATIAL VALIDATION MAP")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(13, 13)
)


roads.plot(
    ax=ax,
    linewidth=0.1,
    alpha=0.15
)


hotspot_grid = (
    grid[
        (
            grid[
                "combined_high_density"
            ] == 1
        )
    ]
    if "combined_high_density" in grid.columns
    else grid[
        grid[
            "collision_density"
        ]
        >= grid[
            "collision_density"
        ]
        .quantile(
            0.95
        )
    ]
)


hotspot_grid.plot(
    ax=ax,
    alpha=0.30,
    edgecolor="none"
)


critical_roads.plot(
    ax=ax,
    linewidth=1.0
)


ax.set_title(
    "Spatial Validation — Critical Roads and Observed Accident Hotspots"
)

ax.set_axis_off()


plt.tight_layout()


validation_map = (
    MAP_DIR /
    "london_spatial_model_validation.png"
)


plt.savefig(
    validation_map,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Spatial validation map saved to:\n"
    f"{validation_map}"
)


# ============================================================
# 16. RISK VS DENSITY SCATTER
# ============================================================

print("\n" + "=" * 80)
print("CREATE RISK VS DENSITY FIGURE")
print("=" * 80)


sample_size = min(
    30000,
    len(joined)
)


plot_sample = (
    joined
    .sample(
        n=sample_size,
        random_state=42
    )
)


fig, ax = plt.subplots(
    figsize=(9, 7)
)


ax.scatter(
    plot_sample[
        "risk_percentile"
    ],
    plot_sample[
        "collision_density"
    ],
    alpha=0.20,
    s=8
)


ax.set_title(
    "Predicted Road Risk vs Observed Collision Density"
)

ax.set_xlabel(
    "Predicted Risk Percentile"
)

ax.set_ylabel(
    "Observed Collision Density"
)


plt.tight_layout()


scatter_file = (
    FIGURE_DIR /
    "predicted_risk_vs_collision_density.png"
)


plt.savefig(
    scatter_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Risk-density figure saved to:\n"
    f"{scatter_file}"
)


# ============================================================
# 17. SAVE VALIDATION SUMMARY
# ============================================================

validation_summary = pd.DataFrame(
    [
        {
            "Risk_Collision_Density_Spearman":
                collision_corr,

            "Risk_Collision_Density_P":
                collision_p,

            "Risk_Severity_Density_Spearman":
                severity_corr,

            "Risk_Severity_Density_P":
                severity_p,

            "Observed_Hotspot_Roads":
                total_hotspots,

            "High_Critical_Hotspot_Overlap":
                high_risk_hotspot_overlap,

            "High_Critical_Hotspot_Overlap_Rate":
                high_risk_hotspot_rate,

            "High_Critical_Hotspot_Capture_Rate":
                hotspot_capture_rate_high,

            "Critical_Hotspot_Overlap":
                critical_hotspot_overlap,

            "Critical_Hotspot_Overlap_Rate":
                critical_hotspot_rate,

            "Critical_Hotspot_Capture_Rate":
                hotspot_capture_rate_critical,

            "High_Critical_Collision_Capture":
                high_collision_capture,

            "High_Critical_Severe_Collision_Capture":
                high_severe_capture,

            "Critical_Collision_Capture":
                critical_collision_capture,

            "Critical_Severe_Collision_Capture":
                critical_severe_capture
        }
    ]
)


validation_summary.to_csv(
    VALIDATION_CSV,
    index=False
)


category_file = (
    OUTPUT_DIR /
    "risk_category_spatial_validation.csv"
)


category_validation.to_csv(
    category_file,
    index=False
)


# ============================================================
# 18. SAVE SPATIAL VALIDATION DATASET
# ============================================================

print("\n" + "=" * 80)
print("SAVE SPATIAL VALIDATION DATASET")
print("=" * 80)


joined.to_file(
    VALIDATION_GPKG,
    layer="spatial_validation",
    driver="GPKG"
)


print(
    f"Validation GeoPackage saved to:\n"
    f"{VALIDATION_GPKG}"
)


print(
    f"\nValidation summary saved to:\n"
    f"{VALIDATION_CSV}"
)


print(
    f"\nCategory validation saved to:\n"
    f"{category_file}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL SPATIAL MODEL VALIDATION CHECK")
print("=" * 80)


print(
    f"Road segments validated: "
    f"{len(joined):,}"
)

print(
    f"Risk vs collision density rho: "
    f"{collision_corr:.4f}"
)

print(
    f"Risk vs severity density rho: "
    f"{severity_corr:.4f}"
)

print(
    f"High/Critical hotspot overlap: "
    f"{high_risk_hotspot_rate:.2f}%"
)

print(
    f"High/Critical hotspot capture: "
    f"{hotspot_capture_rate_high:.2f}%"
)

print(
    f"High/Critical collision capture: "
    f"{high_collision_capture:.2f}%"
)

print(
    f"High/Critical severe collision capture: "
    f"{high_severe_capture:.2f}%"
)

print(
    f"Critical collision capture: "
    f"{critical_collision_capture:.2f}%"
)


print("\n" + "=" * 80)
print("SPATIAL MODEL VALIDATION COMPLETED")
print("=" * 80)