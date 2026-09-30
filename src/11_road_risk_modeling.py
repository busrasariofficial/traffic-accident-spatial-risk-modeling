from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL XGBOOST
# ============================================================

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True

except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "modeling"
)

INPUT_FILE = (
    PROCESSED_DIR /
    "london_road_segment_network_features.csv"
)

RESULTS_FILE = (
    OUTPUT_DIR /
    "road_risk_model_comparison.csv"
)

TEST_PREDICTIONS_FILE = (
    OUTPUT_DIR /
    "road_risk_test_predictions.csv"
)

FEATURE_LIST_FILE = (
    OUTPUT_DIR /
    "road_risk_model_features.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET = "has_collision"

THRESHOLD = 0.50


# ============================================================
# START
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - MACHINE LEARNING")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Modeling dataset not found:\n"
        f"{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)


print("\nDataset loaded successfully.")

print(
    f"\nRows: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
)


# ============================================================
# 2. TARGET VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("TARGET VALIDATION")
print("=" * 80)


if TARGET not in df.columns:

    raise KeyError(
        f"Target variable not found: "
        f"{TARGET}"
    )


df[TARGET] = (
    pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )
)


if df[TARGET].isna().any():

    raise ValueError(
        "Target contains missing values."
    )


df[TARGET] = (
    df[TARGET]
    .astype(int)
)


print(
    df[TARGET]
    .value_counts()
)


print(
    "\nTarget percentages:"
)

print(
    (
        df[TARGET]
        .value_counts(
            normalize=True
        )
        * 100
    )
    .round(2)
)


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

# IMPORTANT:
#
# No collision-derived variables are used as predictors.
#
# Excluded examples:
#
# collision_count
# severe_collision_count
# fatal_collision_count
# collisions_per_km
# severe_collisions_per_km
# has_severe_collision
#
# Coordinates are also excluded from the baseline model
# to reduce direct geographic memorisation.

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


# ============================================================
# 4. FEATURE AVAILABILITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE AVAILABILITY CHECK")
print("=" * 80)


available_numeric = []

for feature in numeric_features:

    if feature in df.columns:

        available_numeric.append(
            feature
        )

        print(
            f"[OK] {feature}"
        )

    else:

        print(
            f"[MISSING] {feature}"
        )


available_categorical = []

for feature in categorical_features:

    if feature in df.columns:

        available_categorical.append(
            feature
        )

        print(
            f"[OK] {feature}"
        )

    else:

        print(
            f"[MISSING] {feature}"
        )


numeric_features = (
    available_numeric
)

categorical_features = (
    available_categorical
)


all_features = (
    numeric_features +
    categorical_features
)


if len(all_features) == 0:

    raise ValueError(
        "No modeling features available."
    )


print(
    f"\nTotal modeling features: "
    f"{len(all_features)}"
)


# ============================================================
# 5. LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("DATA LEAKAGE CHECK")
print("=" * 80)


forbidden_features = {

    "collision_count",
    "severe_collision_count",
    "fatal_collision_count",

    "collisions_per_km",
    "severe_collisions_per_km",

    "has_collision",
    "has_severe_collision",

    "collision_density",
    "severity_density",

    "collision_gi_z",
    "severity_gi_z",

    "priority_cluster",
    "consensus_hotspot"
}


leakage_features = (
    forbidden_features
    .intersection(
        all_features
    )
)


if leakage_features:

    raise ValueError(
        "Potential leakage features detected: "
        f"{leakage_features}"
    )


print(
    "No target-derived leakage features detected."
)


# ============================================================
# 6. PREPARE X AND y
# ============================================================

X = (
    df[
        all_features
    ]
    .copy()
)

y = (
    df[
        TARGET
    ]
    .copy()
)


print("\n" + "=" * 80)
print("MODELING DATASET")
print("=" * 80)


print(
    f"Predictor rows: "
    f"{len(X):,}"
)

print(
    f"Predictor columns: "
    f"{len(X.columns)}"
)

print(
    f"Positive cases: "
    f"{y.sum():,}"
)

print(
    f"Negative cases: "
    f"{(y == 0).sum():,}"
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)


print(
    f"Training rows: "
    f"{len(X_train):,}"
)

print(
    f"Test rows: "
    f"{len(X_test):,}"
)

print(
    f"Training positive rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Test positive rate: "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# 8. CLASS IMBALANCE
# ============================================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count /
    positive_count
)


print("\n" + "=" * 80)
print("CLASS IMBALANCE")
print("=" * 80)


print(
    f"Negative training cases: "
    f"{negative_count:,}"
)

print(
    f"Positive training cases: "
    f"{positive_count:,}"
)

print(
    f"Negative / positive ratio: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# 9. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 10. MODEL DEFINITIONS
# ============================================================

print("\n" + "=" * 80)
print("MODEL DEFINITIONS")
print("=" * 80)


models = {}


# ------------------------------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------------------------------

logistic_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)


models[
    "Logistic Regression"
] = logistic_model


print(
    "Logistic Regression configured."
)


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

random_forest_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]
)


models[
    "Random Forest"
] = random_forest_model


print(
    "Random Forest configured."
)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

if XGBOOST_AVAILABLE:

    xgb_model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                XGBClassifier(

                    n_estimators=400,

                    max_depth=5,

                    learning_rate=0.05,

                    subsample=0.85,

                    colsample_bytree=0.85,

                    scale_pos_weight=scale_pos_weight,

                    objective="binary:logistic",

                    eval_metric="logloss",

                    random_state=RANDOM_STATE,

                    n_jobs=-1
                )
            )
        ]
    )


    models[
        "XGBoost"
    ] = xgb_model


    print(
        "XGBoost configured."
    )


else:

    print(
        "XGBoost is not installed."
    )

    print(
        "The script will continue with "
        "Logistic Regression and Random Forest."
    )


# ============================================================
# 11. MODEL TRAINING FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 80)

    print(
        f"TRAINING MODEL: "
        f"{model_name}"
    )

    print("=" * 80)


    model.fit(
        X_train,
        y_train
    )


    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )


    print(
        f"Accuracy:  "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{pr_auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )


    results = {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "PR_AUC": pr_auc,

        "Threshold": THRESHOLD
    }


    return (
        results,
        probabilities,
        predictions
    )


# ============================================================
# 12. TRAIN ALL MODELS
# ============================================================

all_results = []

prediction_data = pd.DataFrame(

    {

        "actual": (
            y_test
            .reset_index(
                drop=True
            )
        )

    }
)


trained_models = {}


for model_name, model in models.items():

    (
        result,
        probabilities,
        predictions

    ) = evaluate_model(

        model_name,

        model,

        X_train,
        y_train,

        X_test,
        y_test
    )


    all_results.append(
        result
    )


    trained_models[
        model_name
    ] = model


    safe_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    prediction_data[
        f"{safe_name}_probability"
    ] = probabilities


    prediction_data[
        f"{safe_name}_prediction"
    ] = predictions


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)


results_df = pd.DataFrame(
    all_results
)


results_df = (
    results_df
    .sort_values(
        [
            "PR_AUC",
            "ROC_AUC"
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print(
    results_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0][
        "Model"
    ]
)


best_model = (
    trained_models[
        best_model_name
    ]
)


print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)


print(
    f"Selected model based primarily "
    f"on PR-AUC: "
    f"{best_model_name}"
)


best_row = (
    results_df
    .iloc[0]
)


print(
    f"PR-AUC: "
    f"{best_row['PR_AUC']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{best_row['ROC_AUC']:.4f}"
)

print(
    f"Recall: "
    f"{best_row['Recall']:.4f}"
)

print(
    f"Precision: "
    f"{best_row['Precision']:.4f}"
)

print(
    f"F1: "
    f"{best_row['F1']:.4f}"
)


# ============================================================
# 15. SAVE MODELS
# ============================================================

print("\n" + "=" * 80)
print("SAVE TRAINED MODELS")
print("=" * 80)


for model_name, model in trained_models.items():

    safe_name = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    model_path = (
        MODEL_DIR /
        f"{safe_name}.joblib"
    )


    joblib.dump(
        model,
        model_path
    )


    print(
        f"{model_name} saved to:\n"
        f"{model_path}"
    )


# ============================================================
# 16. SAVE BEST MODEL
# ============================================================

best_model_path = (
    MODEL_DIR /
    "best_road_risk_model.joblib"
)


joblib.dump(
    best_model,
    best_model_path
)


print(
    f"\nBest model saved to:\n"
    f"{best_model_path}"
)


# ============================================================
# 17. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


print(
    f"\nModel comparison saved to:\n"
    f"{RESULTS_FILE}"
)


# ============================================================
# 18. SAVE TEST PREDICTIONS
# ============================================================

prediction_data.to_csv(
    TEST_PREDICTIONS_FILE,
    index=False
)


print(
    f"\nTest predictions saved to:\n"
    f"{TEST_PREDICTIONS_FILE}"
)


# ============================================================
# 19. SAVE FEATURE LIST
# ============================================================

feature_list_df = pd.DataFrame(

    {

        "Feature": all_features,

        "Feature_Type": [

            (
                "Numeric"
                if feature in numeric_features
                else "Categorical"
            )

            for feature in all_features
        ]
    }
)


feature_list_df.to_csv(
    FEATURE_LIST_FILE,
    index=False
)


print(
    f"\nFeature list saved to:\n"
    f"{FEATURE_LIST_FILE}"
)


# ============================================================
# 20. BASIC MODEL COMPARISON FIGURE
# ============================================================

figure_file = (
    OUTPUT_DIR /
    "model_comparison.png"
)


plot_df = (
    results_df
    .set_index(
        "Model"
    )
    [
        [
            "PR_AUC",
            "ROC_AUC",
            "Recall",
            "Precision",
            "F1"
        ]
    ]
)


ax = plot_df.plot(
    kind="bar",
    figsize=(11, 7)
)


ax.set_title(
    "Road Segment Collision Risk — Model Comparison"
)

ax.set_xlabel(
    "Model"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\nModel comparison figure saved to:\n"
    f"{figure_file}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL MACHINE LEARNING CHECK")
print("=" * 80)


print(
    f"Training records: "
    f"{len(X_train):,}"
)

print(
    f"Test records: "
    f"{len(X_test):,}"
)

print(
    f"Models trained: "
    f"{len(trained_models)}"
)

print(
    f"Features used: "
    f"{len(all_features)}"
)

print(
    f"Best model: "
    f"{best_model_name}"
)

print(
    f"Best PR-AUC: "
    f"{best_row['PR_AUC']:.4f}"
)

print(
    f"Best ROC-AUC: "
    f"{best_row['ROC_AUC']:.4f}"
)


print("\n" + "=" * 80)
print("ROAD RISK MODELING COMPLETED")
print("=" * 80)