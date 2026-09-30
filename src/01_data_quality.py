from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILE_NAME = "dft-road-casualty-statistics-collision-2025.csv"
FILE_PATH = RAW_DATA_DIR / FILE_NAME


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - DATA QUALITY REPORT")
print("=" * 80)

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{FILE_PATH}\n"
        f"Place '{FILE_NAME}' inside data/raw/"
    )

df = pd.read_csv(FILE_PATH)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print(f"\nTotal collisions: {len(df):,}")
print(f"Total columns: {df.shape[1]}")


# ============================================================
# 1. COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 2. FIRST ROWS
# ============================================================

print("\n" + "=" * 80)
print("FIRST 5 ROWS")
print("=" * 80)

print(df.head())


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(df.dtypes)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing_count = df.isna().sum()

missing_percent = (
    df.isna().mean() * 100
).round(3)

missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percent": missing_percent
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
].sort_values(
    "Missing_Percent",
    ascending=False
)

if missing_report.empty:
    print("No missing values detected.")
else:
    print(missing_report)


# ============================================================
# 5. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

duplicate_rows = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_rows:,}")


# ============================================================
# 6. COLLISION ID QUALITY
# ============================================================

print("\n" + "=" * 80)
print("COLLISION ID QUALITY")
print("=" * 80)

print(
    f"Unique collision_index: "
    f"{df['collision_index'].nunique():,}"
)

print(
    f"Duplicated collision_index: "
    f"{df['collision_index'].duplicated().sum():,}"
)

print(
    f"Missing collision_index: "
    f"{df['collision_index'].isna().sum():,}"
)


# ============================================================
# 7. COORDINATE QUALITY
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL COORDINATE QUALITY")
print("=" * 80)

coordinate_columns = [
    "longitude",
    "latitude",
    "location_easting_osgr",
    "location_northing_osgr"
]

for column in coordinate_columns:

    missing = df[column].isna().sum()

    print(
        f"{column}: "
        f"{missing:,} missing "
        f"({missing / len(df) * 100:.3f}%)"
    )


print("\nLongitude range:")

print(
    f"{df['longitude'].min():.6f} "
    f"to "
    f"{df['longitude'].max():.6f}"
)


print("\nLatitude range:")

print(
    f"{df['latitude'].min():.6f} "
    f"to "
    f"{df['latitude'].max():.6f}"
)


# ============================================================
# 8. INVALID COORDINATES
# ============================================================

print("\n" + "=" * 80)
print("INVALID COORDINATE CHECK")
print("=" * 80)

invalid_longitude = (
    (df["longitude"] < -180)
    | (df["longitude"] > 180)
)

invalid_latitude = (
    (df["latitude"] < -90)
    | (df["latitude"] > 90)
)

zero_coordinates = (
    (df["longitude"] == 0)
    & (df["latitude"] == 0)
)

print(
    f"Invalid longitude values: "
    f"{invalid_longitude.sum():,}"
)

print(
    f"Invalid latitude values: "
    f"{invalid_latitude.sum():,}"
)

print(
    f"Zero coordinate pairs: "
    f"{zero_coordinates.sum():,}"
)


# ============================================================
# 9. COLLISION SEVERITY
# ============================================================

print("\n" + "=" * 80)
print("COLLISION SEVERITY DISTRIBUTION")
print("=" * 80)

severity_counts = (
    df["collision_severity"]
    .value_counts(dropna=False)
    .sort_index()
)

severity_percent = (
    df["collision_severity"]
    .value_counts(
        normalize=True,
        dropna=False
    )
    .sort_index()
    * 100
).round(2)

severity_report = pd.DataFrame({
    "Count": severity_counts,
    "Percent": severity_percent
})

print(severity_report)

print(
    "\nNOTE: collision_severity is coded in the source dataset. "
    "The official code definitions will be decoded during "
    "data preparation."
)


# ============================================================
# 10. ENHANCED SEVERITY
# ============================================================

print("\n" + "=" * 80)
print("ENHANCED COLLISION SEVERITY")
print("=" * 80)

print(
    df["enhanced_severity_collision"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 11. NUMBER OF VEHICLES
# ============================================================

print("\n" + "=" * 80)
print("NUMBER OF VEHICLES")
print("=" * 80)

print(
    df["number_of_vehicles"]
    .describe()
    .round(2)
)

print(
    f"\nZero vehicles: "
    f"{(df['number_of_vehicles'] == 0).sum():,}"
)


# ============================================================
# 12. NUMBER OF CASUALTIES
# ============================================================

print("\n" + "=" * 80)
print("NUMBER OF CASUALTIES")
print("=" * 80)

print(
    df["number_of_casualties"]
    .describe()
    .round(2)
)

print(
    f"\nZero casualties: "
    f"{(df['number_of_casualties'] == 0).sum():,}"
)


# ============================================================
# 13. SPEED LIMIT
# ============================================================

print("\n" + "=" * 80)
print("SPEED LIMIT DISTRIBUTION")
print("=" * 80)

print(
    df["speed_limit"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 14. ROAD TYPE
# ============================================================

print("\n" + "=" * 80)
print("ROAD TYPE DISTRIBUTION")
print("=" * 80)

print(
    df["road_type"]
    .value_counts(dropna=False)
    .sort_index()
)

print(
    "\nNOTE: Road type values are currently coded. "
    "They will be decoded in the preparation stage."
)


# ============================================================
# 15. URBAN / RURAL
# ============================================================

print("\n" + "=" * 80)
print("URBAN / RURAL DISTRIBUTION")
print("=" * 80)

print(
    df["urban_or_rural_area"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 16. LIGHT CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("LIGHT CONDITIONS")
print("=" * 80)

print(
    df["light_conditions"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 17. WEATHER CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("WEATHER CONDITIONS")
print("=" * 80)

print(
    df["weather_conditions"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 18. ROAD SURFACE CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("ROAD SURFACE CONDITIONS")
print("=" * 80)

print(
    df["road_surface_conditions"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 19. JUNCTION DETAIL
# ============================================================

print("\n" + "=" * 80)
print("JUNCTION DETAIL")
print("=" * 80)

print(
    df["junction_detail"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 20. DAY OF WEEK
# ============================================================

print("\n" + "=" * 80)
print("DAY OF WEEK DISTRIBUTION")
print("=" * 80)

print(
    df["day_of_week"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 21. DATE QUALITY
# ============================================================

print("\n" + "=" * 80)
print("DATE QUALITY")
print("=" * 80)

date_parsed = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y",
    errors="coerce"
)

print(
    f"Invalid dates: "
    f"{date_parsed.isna().sum():,}"
)

print(
    f"Minimum date: "
    f"{date_parsed.min()}"
)

print(
    f"Maximum date: "
    f"{date_parsed.max()}"
)


# ============================================================
# 22. TIME QUALITY
# ============================================================

print("\n" + "=" * 80)
print("TIME QUALITY")
print("=" * 80)

time_parsed = pd.to_datetime(
    df["time"],
    format="%H:%M",
    errors="coerce"
)

print(
    f"Missing raw time values: "
    f"{df['time'].isna().sum():,}"
)

print(
    f"Invalid time values: "
    f"{time_parsed.isna().sum():,}"
)


# ============================================================
# 23. COLLISIONS BY MONTH
# ============================================================

print("\n" + "=" * 80)
print("COLLISIONS BY MONTH")
print("=" * 80)

month_counts = (
    date_parsed
    .dt.month
    .value_counts()
    .sort_index()
)

print(month_counts)


# ============================================================
# 24. COLLISIONS BY HOUR
# ============================================================

print("\n" + "=" * 80)
print("COLLISIONS BY HOUR")
print("=" * 80)

hour_counts = (
    time_parsed
    .dt.hour
    .value_counts()
    .sort_index()
)

print(hour_counts)


# ============================================================
# 25. NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("KEY NUMERIC SUMMARY")
print("=" * 80)

numeric_columns = [
    "number_of_vehicles",
    "number_of_casualties",
    "speed_limit",
    "longitude",
    "latitude"
]

print(
    df[numeric_columns]
    .describe()
    .round(3)
)


# ============================================================
# 26. POTENTIAL SENTINEL / UNKNOWN VALUES
# ============================================================

print("\n" + "=" * 80)
print("POTENTIAL CODED UNKNOWN VALUES")
print("=" * 80)

coded_columns = [
    "collision_severity",
    "road_type",
    "speed_limit",
    "junction_detail",
    "junction_control",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area"
]

for column in coded_columns:

    negative_count = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        ) < 0
    ).sum()

    print(
        f"{column}: "
        f"{negative_count:,} negative-coded values"
    )


# ============================================================
# 27. SPATIAL ANALYSIS READINESS
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL ANALYSIS READINESS")
print("=" * 80)

valid_spatial = (
    df["longitude"].notna()
    & df["latitude"].notna()
    & ~invalid_longitude
    & ~invalid_latitude
    & ~zero_coordinates
)

valid_spatial_count = valid_spatial.sum()

print(
    f"Valid spatial records: "
    f"{valid_spatial_count:,}"
)

print(
    f"Invalid / missing spatial records: "
    f"{len(df) - valid_spatial_count:,}"
)

print(
    f"Spatial usability rate: "
    f"{valid_spatial_count / len(df) * 100:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY CHECK COMPLETED")
print("=" * 80)

print(f"""
Dataset:
{FILE_NAME}

Rows:
{len(df):,}

Columns:
{df.shape[1]}

Unique collisions:
{df['collision_index'].nunique():,}

Duplicate rows:
{duplicate_rows:,}

Valid spatial records:
{valid_spatial_count:,}
""")