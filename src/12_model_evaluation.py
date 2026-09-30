from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

INPUT_FILE = (
    PROCESSED_DIR /
    "london_road_segment_network_features.csv"
)

LOGISTIC_MODEL_FILE = (
    MODEL_DIR /
    "logistic_regression.joblib"
)

RF_MODEL_FILE = (
    MODEL_DIR /
    "random_forest.joblib"
)

THRESHOLD_RESULTS_FILE = (
    OUTPUT_DIR /
    "threshold_evaluation.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR /
    "feature_importance.csv"
)

FINAL_METRICS_FILE = (
    OUTPUT_DIR /
    "final_model_evaluation.csv"
)

SELECTED_THRESHOLD_FILE = (
    OUTPUT_DIR /
    "selected_threshold.txt"
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
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "has_collision"


# ============================================================
# START
# ============================================================

print("=" * 80)
print("TRAFFIC ACCIDENT SPATIAL RISK MODELING - MODEL EVALUATION")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nDataset loaded successfully.")

print(
    f"Rows: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
)


# ============================================================
# 2. MODEL FEATURES
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
    .astype(int)
)


# ============================================================
# 3. RECREATE TEST SPLIT
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
print("TEST DATASET")
print("=" * 80)

print(
    f"Test records: "
    f"{len(X_test):,}"
)

print(
    f"Positive cases: "
    f"{y_test.sum():,}"
)

print(
    f"Positive rate: "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

print("\n" + "=" * 80)
print("LOAD TRAINED MODELS")
print("=" * 80)


logistic_model = joblib.load(
    LOGISTIC_MODEL_FILE
)

random_forest_model = joblib.load(
    RF_MODEL_FILE
)


print("Logistic Regression loaded.")
print("Random Forest loaded.")


# ============================================================
# 5. PREDICT PROBABILITIES
# ============================================================

logistic_prob = (
    logistic_model
    .predict_proba(
        X_test
    )[:, 1]
)

rf_prob = (
    random_forest_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 6. GLOBAL DISCRIMINATION METRICS
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MODEL PERFORMANCE")
print("=" * 80)


model_probabilities = {

    "Logistic Regression":
        logistic_prob,

    "Random Forest":
        rf_prob
}


global_results = []


for model_name, probability in model_probabilities.items():

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    pr_auc = average_precision_score(
        y_test,
        probability
    )

    global_results.append(
        {
            "Model": model_name,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        }
    )


global_df = pd.DataFrame(
    global_results
)


print(
    global_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 7. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION - THRESHOLD ANALYSIS")
print("=" * 80)


thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        logistic_prob >= threshold
    ).astype(int)


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


    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions
        )
        .ravel()
    )


    specificity = (
        tn /
        (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    balanced_accuracy = (
        recall +
        specificity
    ) / 2


    threshold_results.append(
        {
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "Balanced_Accuracy": balanced_accuracy,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print(
    threshold_df[
        [
            "Threshold",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "Balanced_Accuracy"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 8. SELECT THRESHOLD
# ============================================================

# Primary rule:
# choose the threshold with the highest F1 score.
#
# This balances precision and recall while retaining
# useful sensitivity to the minority collision class.

best_threshold_row = (
    threshold_df
    .sort_values(
        [
            "F1",
            "Recall"
        ],
        ascending=False
    )
    .iloc[0]
)


selected_threshold = (
    float(
        best_threshold_row[
            "Threshold"
        ]
    )
)


print("\n" + "=" * 80)
print("SELECTED OPERATING THRESHOLD")
print("=" * 80)


print(
    f"Selected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_threshold_row['Precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_threshold_row['Recall']:.4f}"
)

print(
    f"F1: "
    f"{best_threshold_row['F1']:.4f}"
)

print(
    f"Specificity: "
    f"{best_threshold_row['Specificity']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_threshold_row['Balanced_Accuracy']:.4f}"
)


# ============================================================
# 9. FINAL LOGISTIC PREDICTIONS
# ============================================================

final_predictions = (
    logistic_prob >= selected_threshold
).astype(int)


final_precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    y_test,
    logistic_prob
)

final_pr_auc = average_precision_score(
    y_test,
    logistic_prob
)


tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        final_predictions
    )
    .ravel()
)


final_specificity = (
    tn /
    (tn + fp)
)


final_balanced_accuracy = (
    final_recall +
    final_specificity
) / 2


print("\n" + "=" * 80)
print("FINAL LOGISTIC REGRESSION PERFORMANCE")
print("=" * 80)


print(
    f"Threshold:          "
    f"{selected_threshold:.2f}"
)

print(
    f"Precision:          "
    f"{final_precision:.4f}"
)

print(
    f"Recall:             "
    f"{final_recall:.4f}"
)

print(
    f"F1 Score:           "
    f"{final_f1:.4f}"
)

print(
    f"Specificity:        "
    f"{final_specificity:.4f}"
)

print(
    f"Balanced Accuracy:  "
    f"{final_balanced_accuracy:.4f}"
)

print(
    f"ROC-AUC:            "
    f"{final_roc_auc:.4f}"
)

print(
    f"PR-AUC:             "
    f"{final_pr_auc:.4f}"
)


print("\nConfusion Matrix:")

print(
    np.array(
        [
            [tn, fp],
            [fn, tp]
        ]
    )
)


# ============================================================
# 10. FEATURE NAMES
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
print("=" * 80)


preprocessor = (
    logistic_model
    .named_steps[
        "preprocessor"
    ]
)


model = (
    logistic_model
    .named_steps[
        "model"
    ]
)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


coefficients = (
    model
    .coef_[0]
)


feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Coefficient":
            np.abs(coefficients),
        "Odds_Ratio":
            np.exp(coefficients)
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "Absolute_Coefficient",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print(
    feature_importance
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. ROC CURVE
# ============================================================

print("\n" + "=" * 80)
print("CREATE ROC CURVE")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


for model_name, probability in model_probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc_value = roc_auc_score(
        y_test,
        probability
    )

    ax.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"(AUC={auc_value:.3f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)


ax.set_title(
    "ROC Curve — Road Segment Collision Risk"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.legend()

plt.tight_layout()


roc_file = (
    FIGURE_DIR /
    "road_risk_roc_curve.png"
)


plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"ROC curve saved to:\n"
    f"{roc_file}"
)


# ============================================================
# 12. PRECISION-RECALL CURVE
# ============================================================

print("\n" + "=" * 80)
print("CREATE PRECISION-RECALL CURVE")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


for model_name, probability in model_probabilities.items():

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test,
            probability
        )
    )

    pr_auc = average_precision_score(
        y_test,
        probability
    )

    ax.plot(
        recall_curve,
        precision_curve,
        label=(
            f"{model_name} "
            f"(AP={pr_auc:.3f})"
        )
    )


baseline = y_test.mean()


ax.axhline(
    baseline,
    linestyle="--",
    label=(
        f"Baseline prevalence "
        f"({baseline:.3f})"
    )
)


ax.set_title(
    "Precision-Recall Curve — Road Segment Collision Risk"
)

ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.legend()

plt.tight_layout()


pr_file = (
    FIGURE_DIR /
    "road_risk_precision_recall_curve.png"
)


plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Precision-Recall curve saved to:\n"
    f"{pr_file}"
)


# ============================================================
# 13. THRESHOLD PERFORMANCE FIGURE
# ============================================================

print("\n" + "=" * 80)
print("CREATE THRESHOLD PERFORMANCE FIGURE")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.plot(
    threshold_df[
        "Threshold"
    ],
    threshold_df[
        "Precision"
    ],
    marker="o",
    label="Precision"
)

ax.plot(
    threshold_df[
        "Threshold"
    ],
    threshold_df[
        "Recall"
    ],
    marker="o",
    label="Recall"
)

ax.plot(
    threshold_df[
        "Threshold"
    ],
    threshold_df[
        "F1"
    ],
    marker="o",
    label="F1"
)


ax.axvline(
    selected_threshold,
    linestyle="--",
    label=(
        f"Selected threshold "
        f"({selected_threshold:.2f})"
    )
)


ax.set_title(
    "Threshold Performance — Logistic Regression"
)

ax.set_xlabel(
    "Probability Threshold"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

ax.legend()

plt.tight_layout()


threshold_figure = (
    FIGURE_DIR /
    "road_risk_threshold_performance.png"
)


plt.savefig(
    threshold_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Threshold figure saved to:\n"
    f"{threshold_figure}"
)


# ============================================================
# 14. FEATURE IMPORTANCE FIGURE
# ============================================================

print("\n" + "=" * 80)
print("CREATE FEATURE IMPORTANCE FIGURE")
print("=" * 80)


top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "Coefficient"
    )
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


ax.barh(
    top_features[
        "Feature"
    ],
    top_features[
        "Coefficient"
    ]
)


ax.axvline(
    0,
    linewidth=1
)


ax.set_title(
    "Top Logistic Regression Coefficients"
)

ax.set_xlabel(
    "Standardized Model Coefficient"
)

ax.set_ylabel(
    "Feature"
)


plt.tight_layout()


importance_figure = (
    FIGURE_DIR /
    "road_risk_feature_importance.png"
)


plt.savefig(
    importance_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Feature importance figure saved to:\n"
    f"{importance_figure}"
)


# ============================================================
# 15. SAVE THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    THRESHOLD_RESULTS_FILE,
    index=False
)


# ============================================================
# 16. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)


# ============================================================
# 17. SAVE SELECTED THRESHOLD
# ============================================================

with open(
    SELECTED_THRESHOLD_FILE,
    "w"
) as file:

    file.write(
        str(
            selected_threshold
        )
    )


# ============================================================
# 18. SAVE FINAL METRICS
# ============================================================

final_metrics = pd.DataFrame(
    [
        {
            "Model":
                "Logistic Regression",

            "Threshold":
                selected_threshold,

            "Precision":
                final_precision,

            "Recall":
                final_recall,

            "F1":
                final_f1,

            "Specificity":
                final_specificity,

            "Balanced_Accuracy":
                final_balanced_accuracy,

            "ROC_AUC":
                final_roc_auc,

            "PR_AUC":
                final_pr_auc,

            "True_Positive":
                tp,

            "False_Positive":
                fp,

            "True_Negative":
                tn,

            "False_Negative":
                fn
        }
    ]
)


final_metrics.to_csv(
    FINAL_METRICS_FILE,
    index=False
)


print("\n" + "=" * 80)
print("SAVE EVALUATION OUTPUTS")
print("=" * 80)


print(
    f"Threshold results:\n"
    f"{THRESHOLD_RESULTS_FILE}"
)

print(
    f"\nFeature importance:\n"
    f"{FEATURE_IMPORTANCE_FILE}"
)

print(
    f"\nSelected threshold:\n"
    f"{SELECTED_THRESHOLD_FILE}"
)

print(
    f"\nFinal metrics:\n"
    f"{FINAL_METRICS_FILE}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL EVALUATION CHECK")
print("=" * 80)


print(
    f"Selected model: "
    f"Logistic Regression"
)

print(
    f"Selected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"ROC-AUC: "
    f"{final_roc_auc:.4f}"
)

print(
    f"PR-AUC: "
    f"{final_pr_auc:.4f}"
)

print(
    f"Precision: "
    f"{final_precision:.4f}"
)

print(
    f"Recall: "
    f"{final_recall:.4f}"
)

print(
    f"F1: "
    f"{final_f1:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{final_balanced_accuracy:.4f}"
)


print("\n" + "=" * 80)
print("MODEL EVALUATION COMPLETED")
print("=" * 80)