from pathlib import Path
import pandas as pd
import geopandas as gpd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "traffic_collisions_2025_clean.csv"
)

OUTPUT_GPKG = (
    PROCESSED_DATA_DIR /
    "traffic_collisions_2025_spatial.gpkg"
)

OUTPUT_CSV = (
    PROCESSED_DATA_DIR /
    "traffic_collisions_2025_spatial.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - SPATIAL PREPARATION")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nDataset loaded successfully.")

print(f"\nRows: {len(df):,}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 1. COORDINATE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("COORDINATE VALIDATION")
print("=" * 80)

missing_longitude = df["longitude"].isna().sum()
missing_latitude = df["latitude"].isna().sum()

invalid_longitude = (
    ~df["longitude"].between(-180, 180)
).sum()

invalid_latitude = (
    ~df["latitude"].between(-90, 90)
).sum()

zero_coordinates = (
    (df["longitude"] == 0)
    & (df["latitude"] == 0)
).sum()

print(f"Missing longitude: {missing_longitude:,}")
print(f"Missing latitude: {missing_latitude:,}")
print(f"Invalid longitude: {invalid_longitude:,}")
print(f"Invalid latitude: {invalid_latitude:,}")
print(f"Zero coordinate pairs: {zero_coordinates:,}")


# ============================================================
# 2. CREATE GEODATAFRAME
# ============================================================

print("\n" + "=" * 80)
print("GEODATAFRAME CREATION")
print("=" * 80)

geometry = gpd.points_from_xy(
    df["longitude"],
    df["latitude"]
)

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4326"
)

print("GeoDataFrame created successfully.")

print(f"Geometry type: {gdf.geometry.geom_type.unique()}")
print(f"Initial CRS: {gdf.crs}")

print(
    f"Missing geometries: "
    f"{gdf.geometry.isna().sum():,}"
)

print(
    f"Empty geometries: "
    f"{gdf.geometry.is_empty.sum():,}"
)


# ============================================================
# 3. GEOMETRY VALIDITY
# ============================================================

print("\n" + "=" * 80)
print("GEOMETRY VALIDATION")
print("=" * 80)

valid_geometry_count = gdf.geometry.is_valid.sum()
invalid_geometry_count = (~gdf.geometry.is_valid).sum()

print(
    f"Valid geometries: "
    f"{valid_geometry_count:,}"
)

print(
    f"Invalid geometries: "
    f"{invalid_geometry_count:,}"
)


# ============================================================
# 4. WGS84 SPATIAL EXTENT
# ============================================================

print("\n" + "=" * 80)
print("WGS84 SPATIAL EXTENT")
print("=" * 80)

minx, miny, maxx, maxy = gdf.total_bounds

print(f"Minimum longitude: {minx:.6f}")
print(f"Minimum latitude: {miny:.6f}")
print(f"Maximum longitude: {maxx:.6f}")
print(f"Maximum latitude: {maxy:.6f}")


# ============================================================
# 5. PROJECT TO BRITISH NATIONAL GRID
# ============================================================

print("\n" + "=" * 80)
print("CRS TRANSFORMATION")
print("=" * 80)

print("Source CRS:")
print(gdf.crs)

gdf_projected = gdf.to_crs(
    epsg=27700
)

print("\nProjected CRS:")
print(gdf_projected.crs)

print(
    "\nCoordinates are now represented in "
    "British National Grid metres."
)


# ============================================================
# 6. PROJECTED COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("PROJECTED COORDINATES")
print("=" * 80)

gdf_projected["bng_easting"] = (
    gdf_projected.geometry.x
)

gdf_projected["bng_northing"] = (
    gdf_projected.geometry.y
)

print(
    gdf_projected[
        [
            "collision_index",
            "longitude",
            "latitude",
            "bng_easting",
            "bng_northing"
        ]
    ]
    .head()
    .to_string(index=False)
)


# ============================================================
# 7. COMPARE WITH SOURCE OSGR COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("SOURCE VS CALCULATED BNG COORDINATES")
print("=" * 80)

comparison_mask = (
    gdf_projected["location_easting_osgr"].notna()
    & gdf_projected["location_northing_osgr"].notna()
)

comparison = (
    gdf_projected
    .loc[comparison_mask]
    .copy()
)

comparison["easting_difference"] = (
    comparison["bng_easting"]
    - comparison["location_easting_osgr"]
).abs()

comparison["northing_difference"] = (
    comparison["bng_northing"]
    - comparison["location_northing_osgr"]
).abs()

print(
    f"Records available for coordinate comparison: "
    f"{len(comparison):,}"
)

print(
    f"Mean absolute easting difference: "
    f"{comparison['easting_difference'].mean():.2f} m"
)

print(
    f"Mean absolute northing difference: "
    f"{comparison['northing_difference'].mean():.2f} m"
)

print(
    f"Maximum easting difference: "
    f"{comparison['easting_difference'].max():.2f} m"
)

print(
    f"Maximum northing difference: "
    f"{comparison['northing_difference'].max():.2f} m"
)


# ============================================================
# 8. PROJECTED SPATIAL EXTENT
# ============================================================

print("\n" + "=" * 80)
print("PROJECTED SPATIAL EXTENT")
print("=" * 80)

min_easting, min_northing, max_easting, max_northing = (
    gdf_projected.total_bounds
)

print(f"Minimum easting: {min_easting:,.2f} m")
print(f"Maximum easting: {max_easting:,.2f} m")

print(f"Minimum northing: {min_northing:,.2f} m")
print(f"Maximum northing: {max_northing:,.2f} m")


# ============================================================
# 9. SIMPLE SPATIAL DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL DISTRIBUTION CHECK")
print("=" * 80)

print("\nUrban / rural:")

print(
    gdf_projected[
        "urban_rural_label"
    ]
    .value_counts(dropna=False)
)

print("\nTop 10 local authorities by collision count:")

top_authorities = (
    gdf_projected[
        "local_authority_district"
    ]
    .value_counts()
    .head(10)
)

print(top_authorities)


# ============================================================
# 10. SPATIAL SEVERITY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL SEVERITY SUMMARY")
print("=" * 80)

severity_summary = (
    gdf_projected
    .groupby(
        "severity_label",
        observed=True
    )
    .agg(
        Collisions=("collision_index", "count"),
        Mean_Easting=("bng_easting", "mean"),
        Mean_Northing=("bng_northing", "mean")
    )
    .round(2)
)

print(severity_summary)


# ============================================================
# 11. SAVE GEOPACKAGE
# ============================================================

print("\n" + "=" * 80)
print("SAVE SPATIAL DATA")
print("=" * 80)

gdf_projected.to_file(
    OUTPUT_GPKG,
    layer="collisions_2025",
    driver="GPKG"
)

print(
    f"GeoPackage saved to:\n"
    f"{OUTPUT_GPKG}"
)


# ============================================================
# 12. SAVE SPATIAL CSV
# ============================================================

# CSV cannot preserve GIS geometry in the same way as
# GeoPackage, so geometry is excluded here.
#
# The projected X/Y coordinates are retained for SQL,
# Power BI and other tabular workflows.

spatial_csv = (
    gdf_projected
    .drop(columns="geometry")
    .copy()
)

spatial_csv.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSpatial CSV saved to:\n"
    f"{OUTPUT_CSV}"
)


# ============================================================
# FINAL SPATIAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL SPATIAL CHECK")
print("=" * 80)

print(f"Spatial records: {len(gdf_projected):,}")

print(
    f"Valid geometries: "
    f"{gdf_projected.geometry.is_valid.sum():,}"
)

print(
    f"Missing geometries: "
    f"{gdf_projected.geometry.isna().sum():,}"
)

print(
    f"CRS: "
    f"{gdf_projected.crs}"
)

print(
    f"Unique collision IDs: "
    f"{gdf_projected['collision_index'].nunique():,}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL PREPARATION COMPLETED")
print("=" * 80)