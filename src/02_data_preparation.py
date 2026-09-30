from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = RAW_DATA_DIR / "dft-road-casualty-statistics-collision-2025.csv"

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "traffic_collisions_2025_clean.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - DATA PREPARATION")
print("=" * 80)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")

print(f"\nOriginal rows: {len(df):,}")
print(f"Original columns: {df.shape[1]}")


# ============================================================
# 1. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE HANDLING")
print("=" * 80)

before = len(df)

df = df.drop_duplicates(
    subset="collision_index"
).copy()

removed_duplicates = before - len(df)

print(
    f"Duplicate collision records removed: "
    f"{removed_duplicates:,}"
)


# ============================================================
# 2. SPATIAL COORDINATE CLEANING
# ============================================================

print("\n" + "=" * 80)
print("SPATIAL COORDINATE CLEANING")
print("=" * 80)

before = len(df)

valid_coordinates = (
    df["longitude"].notna()
    & df["latitude"].notna()
    & df["longitude"].between(-180, 180)
    & df["latitude"].between(-90, 90)
    & ~(
        (df["longitude"] == 0)
        & (df["latitude"] == 0)
    )
)

df = df.loc[valid_coordinates].copy()

removed_spatial = before - len(df)

print(
    f"Records removed due to invalid/missing coordinates: "
    f"{removed_spatial:,}"
)

print(
    f"Spatially valid records remaining: "
    f"{len(df):,}"
)


# ============================================================
# 3. DATE PREPARATION
# ============================================================

print("\n" + "=" * 80)
print("DATE FEATURE ENGINEERING")
print("=" * 80)

df["date"] = pd.to_datetime(
    df["date"],
    format="%d/%m/%Y",
    errors="coerce"
)

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["month_name"] = df["date"].dt.month_name()

df["day_of_month"] = df["date"].dt.day

df["week_of_year"] = (
    df["date"]
    .dt
    .isocalendar()
    .week
    .astype("Int64")
)

df["quarter"] = df["date"].dt.quarter


print(
    f"Invalid dates after conversion: "
    f"{df['date'].isna().sum():,}"
)


# ============================================================
# 4. DAY OF WEEK
# ============================================================

print("\n" + "=" * 80)
print("DAY OF WEEK FEATURES")
print("=" * 80)

# STATS19:
# 1 = Sunday
# 2 = Monday
# ...
# 7 = Saturday

day_map = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}

df["day_name"] = (
    df["day_of_week"]
    .map(day_map)
)

df["is_weekend"] = (
    df["day_of_week"]
    .isin([1, 7])
    .astype(int)
)

print(
    df["day_name"]
    .value_counts()
)


# ============================================================
# 5. TIME PREPARATION
# ============================================================

print("\n" + "=" * 80)
print("TIME FEATURE ENGINEERING")
print("=" * 80)

parsed_time = pd.to_datetime(
    df["time"],
    format="%H:%M",
    errors="coerce"
)

df["hour"] = parsed_time.dt.hour

df["minute"] = parsed_time.dt.minute

print(
    f"Invalid time values: "
    f"{df['hour'].isna().sum():,}"
)


# ============================================================
# 6. TIME PERIOD
# ============================================================

def classify_time_period(hour):

    if pd.isna(hour):
        return "Unknown"

    if 5 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 21:
        return "Evening"

    else:
        return "Night"


df["time_period"] = (
    df["hour"]
    .apply(classify_time_period)
)


# ============================================================
# 7. RUSH HOUR
# ============================================================

# Morning rush:
# 07:00 - 09:59
#
# Evening rush:
# 16:00 - 18:59

df["is_rush_hour"] = (
    (
        df["hour"].between(7, 9)
    )
    |
    (
        df["hour"].between(16, 18)
    )
).astype(int)


print("\nTime periods:")

print(
    df["time_period"]
    .value_counts()
)

print(
    f"\nRush-hour collisions: "
    f"{df['is_rush_hour'].sum():,}"
)


# ============================================================
# 8. COLLISION SEVERITY
# ============================================================

print("\n" + "=" * 80)
print("COLLISION SEVERITY PREPARATION")
print("=" * 80)

severity_map = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

df["severity_label"] = (
    df["collision_severity"]
    .map(severity_map)
)


# Binary ML target:
#
# 1 = Fatal / Serious
# 0 = Slight

df["is_severe"] = (
    df["collision_severity"]
    .isin([1, 2])
    .astype(int)
)


print(
    df["severity_label"]
    .value_counts()
)

print(
    f"\nSevere collisions: "
    f"{df['is_severe'].sum():,}"
)

print(
    f"Severe collision rate: "
    f"{df['is_severe'].mean() * 100:.2f}%"
)


# ============================================================
# 9. ROAD TYPE
# ============================================================

print("\n" + "=" * 80)
print("ROAD TYPE PREPARATION")
print("=" * 80)

road_type_map = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown"
}

df["road_type_label"] = (
    df["road_type"]
    .map(road_type_map)
    .fillna("Unknown")
)

print(
    df["road_type_label"]
    .value_counts()
)


# ============================================================
# 10. URBAN / RURAL AREA
# ============================================================

print("\n" + "=" * 80)
print("URBAN / RURAL PREPARATION")
print("=" * 80)

urban_rural_map = {
    1: "Urban",
    2: "Rural",
    3: "Unallocated"
}

df["urban_rural_label"] = (
    df["urban_or_rural_area"]
    .map(urban_rural_map)
    .fillna("Unknown")
)

print(
    df["urban_rural_label"]
    .value_counts()
)


# ============================================================
# 11. LIGHT CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("LIGHT CONDITIONS PREPARATION")
print("=" * 80)

light_map = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Unknown"
}

df["light_conditions_label"] = (
    df["light_conditions"]
    .map(light_map)
    .fillna("Unknown")
)

df["is_dark"] = (
    df["light_conditions"]
    .isin([4, 5, 6, 7])
    .astype(int)
)

print(
    df["light_conditions_label"]
    .value_counts()
)


# ============================================================
# 12. WEATHER CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("WEATHER CONDITIONS PREPARATION")
print("=" * 80)

weather_map = {
    1: "Fine - no high winds",
    2: "Raining - no high winds",
    3: "Snowing - no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown"
}

df["weather_label"] = (
    df["weather_conditions"]
    .map(weather_map)
    .fillna("Unknown")
)


adverse_weather_codes = [
    2,
    3,
    4,
    5,
    6,
    7
]

df["is_adverse_weather"] = (
    df["weather_conditions"]
    .isin(adverse_weather_codes)
    .astype(int)
)

print(
    df["weather_label"]
    .value_counts()
)


# ============================================================
# 13. ROAD SURFACE CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("ROAD SURFACE PREPARATION")
print("=" * 80)

road_surface_map = {
    1: "Dry",
    2: "Wet or damp",
    3: "Snow",
    4: "Frost or ice",
    5: "Flood over 3cm deep",
    9: "Unknown",
    -1: "Unknown"
}

df["road_surface_label"] = (
    df["road_surface_conditions"]
    .map(road_surface_map)
    .fillna("Unknown")
)


df["is_adverse_surface"] = (
    df["road_surface_conditions"]
    .isin([2, 3, 4, 5])
    .astype(int)
)

print(
    df["road_surface_label"]
    .value_counts()
)


# ============================================================
# 14. SPEED LIMIT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("SPEED LIMIT FEATURES")
print("=" * 80)

df["speed_limit"] = pd.to_numeric(
    df["speed_limit"],
    errors="coerce"
)


def speed_band(speed):

    if pd.isna(speed):
        return "Unknown"

    if speed <= 20:
        return "20 or less"

    elif speed <= 30:
        return "30"

    elif speed <= 40:
        return "40"

    elif speed <= 50:
        return "50"

    else:
        return "60+"


df["speed_band"] = (
    df["speed_limit"]
    .apply(speed_band)
)


# DfT commonly treats <=40 mph as built-up
# for road environment reporting.

df["is_high_speed_road"] = (
    df["speed_limit"] > 40
).astype(int)


print(
    df["speed_band"]
    .value_counts()
)


# ============================================================
# 15. JUNCTION FEATURES
# ============================================================

print("\n" + "=" * 80)
print("JUNCTION FEATURES")
print("=" * 80)

# The 2025 data contains records collected under
# both the previous and new STATS19 specifications.
#
# Therefore, instead of forcing every junction code
# into an old textual mapping, we preserve the original
# junction_detail value and create robust derived features.

df["junction_detail_clean"] = (
    df["junction_detail"]
    .replace({
        -1: np.nan,
        99: np.nan
    })
)


# 0 represents collisions not at / within the
# junction definition in the harmonised data.

df["is_at_junction"] = np.where(
    df["junction_detail_clean"].isna(),
    np.nan,
    (
        df["junction_detail_clean"] != 0
    ).astype(int)
)


print(
    "Original junction codes:"
)

print(
    df["junction_detail"]
    .value_counts()
    .sort_index()
)


print(
    f"\nKnown junction records: "
    f"{df['junction_detail_clean'].notna().sum():,}"
)


# ============================================================
# 16. UNKNOWN CODE HANDLING
# ============================================================

print("\n" + "=" * 80)
print("UNKNOWN / NO-DATA HANDLING")
print("=" * 80)

coded_columns = [
    "junction_detail",
    "junction_control",
    "light_conditions",
    "road_surface_conditions"
]

for column in coded_columns:

    unknown_count = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        ) < 0
    ).sum()

    print(
        f"{column}: "
        f"{unknown_count:,} negative/no-data values preserved "
        f"as source codes"
    )


# ============================================================
# 17. CASUALTY FEATURES
# ============================================================

print("\n" + "=" * 80)
print("CASUALTY FEATURES")
print("=" * 80)

df["multiple_casualties"] = (
    df["number_of_casualties"] > 1
).astype(int)

df["multiple_vehicles"] = (
    df["number_of_vehicles"] > 1
).astype(int)


print(
    f"Multiple-casualty collisions: "
    f"{df['multiple_casualties'].sum():,}"
)

print(
    f"Multiple-vehicle collisions: "
    f"{df['multiple_vehicles'].sum():,}"
)


# ============================================================
# 18. TEMPORAL COMBINATION
# ============================================================

print("\n" + "=" * 80)
print("DATETIME CREATION")
print("=" * 80)

df["collision_datetime"] = pd.to_datetime(
    (
        df["date"].dt.strftime("%Y-%m-%d")
        + " "
        + df["time"].astype(str)
    ),
    errors="coerce"
)

print(
    f"Valid collision datetimes: "
    f"{df['collision_datetime'].notna().sum():,}"
)


# ============================================================
# 19. FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ENGINEERED FEATURES")
print("=" * 80)

engineered_features = [
    "year",
    "month",
    "month_name",
    "day_of_month",
    "week_of_year",
    "quarter",
    "day_name",
    "is_weekend",
    "hour",
    "minute",
    "time_period",
    "is_rush_hour",
    "severity_label",
    "is_severe",
    "road_type_label",
    "urban_rural_label",
    "light_conditions_label",
    "is_dark",
    "weather_label",
    "is_adverse_weather",
    "road_surface_label",
    "is_adverse_surface",
    "speed_band",
    "is_high_speed_road",
    "junction_detail_clean",
    "is_at_junction",
    "multiple_casualties",
    "multiple_vehicles",
    "collision_datetime"
]

for feature in engineered_features:
    print(feature)


# ============================================================
# 20. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL PREPARATION CHECK")
print("=" * 80)

print(
    f"Rows: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{df.shape[1]}"
)

print(
    f"Duplicate collision IDs: "
    f"{df['collision_index'].duplicated().sum():,}"
)

print(
    f"Missing longitude: "
    f"{df['longitude'].isna().sum():,}"
)

print(
    f"Missing latitude: "
    f"{df['latitude'].isna().sum():,}"
)

print(
    f"Missing severity labels: "
    f"{df['severity_label'].isna().sum():,}"
)

print(
    f"Missing dates: "
    f"{df['date'].isna().sum():,}"
)

print(
    f"Missing hours: "
    f"{df['hour'].isna().sum():,}"
)


# ============================================================
# 21. SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 80)
print("SAVE PROCESSED DATASET")
print("=" * 80)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Processed dataset saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETED")
print("=" * 80)