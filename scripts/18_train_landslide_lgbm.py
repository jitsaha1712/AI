
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. Paths
# ============================================================

DATASET_CSV = (
    r"C:\NERProject\Data\processed"
    r"\landslide_ml_dataset.csv"
)

MODEL_DIR = (
    r"C:\NERProject\models"
)

MODEL_PATH = (
    r"C:\NERProject\models"
    r"\lgbm_landslide.pkl"
)

CV_RESULTS_PATH = (
    r"C:\NERProject\Data\processed"
    r"\landslide_model_cv_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    r"C:\NERProject\Data\processed"
    r"\landslide_feature_importance.csv"
)


# ============================================================
# 2. Create model directory
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. Random seed
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 4. Features
# ============================================================
#
# rain_1d is intentionally removed because:
#
# rain_1d == rain_event_day
#
# Longitude/latitude are also not used because we don't want
# the model to simply memorize geographical locations.
#
# event_date is used only for grouping during validation.
#

FEATURE_COLUMNS = [
    "rain_event_day",
    "rain_3d",
    "rain_7d",
    "rain_10d",
    "max_rain_3d",
    "max_rain_7d",
    "heavy_rain_days",
    "elevation_m",
    "slope_deg"
]


# ============================================================
# 5. Read dataset
# ============================================================

print("Reading ML dataset...")

df = pd.read_csv(
    DATASET_CSV
)

print(
    "Samples:",
    len(df)
)


# ============================================================
# 6. Basic validation
# ============================================================

print("\nChecking dataset...")

required_columns = (
    FEATURE_COLUMNS
    +
    [
        "label",
        "event_date"
    ]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if len(missing_columns) > 0:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


if df[FEATURE_COLUMNS].isna().any().any():

    raise ValueError(
        "Missing feature values found."
    )


if not np.isfinite(
    df[FEATURE_COLUMNS].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Invalid numeric values found."
    )


# ============================================================
# 7. Prepare X and y
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    "label"
].astype(int)


# ============================================================
# 8. Prepare groups
# ============================================================
#
# Control points can share the same historical date as
# landslide events.
#
# Therefore, during validation we keep the same date entirely
# inside either training or validation.
#
# This reduces leakage from shared rainfall conditions.
#

groups = pd.to_datetime(
    df["event_date"]
).dt.strftime(
    "%Y-%m-%d"
)


# ============================================================
# 9. Class statistics
# ============================================================

positive_count = int(
    (y == 1).sum()
)

negative_count = int(
    (y == 0).sum()
)


scale_pos_weight = (
    negative_count
    /
    positive_count
)


print(
    "\nClass distribution:"
)

print(
    "Positive:",
    positive_count
)

print(
    "Control:",
    negative_count
)

print(
    "Scale positive weight:",
    round(
        scale_pos_weight,
        4
    )
)


# ============================================================
# 10. LightGBM parameters
# ============================================================

model_params = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,

    "num_leaves": 15,
    "max_depth": 4,

    "min_child_samples": 10,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.1,
    "reg_lambda": 1.0,

    "scale_pos_weight": scale_pos_weight,

    "random_state": RANDOM_SEED,
    "n_jobs": -1,

    "verbosity": -1
}


# ============================================================
# 11. Cross-validation
# ============================================================

print(
    "\nStarting 5-fold grouped cross-validation..."
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)


fold_results = []


all_actual = []

all_probabilities = []

all_predictions = []


for fold, (
    train_index,
    valid_index
) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print(
        "\n----------------------------------------"
    )

    print(
        "Fold",
        fold
    )

    print(
        "----------------------------------------"
    )


    X_train = X.iloc[
        train_index
    ]

    X_valid = X.iloc[
        valid_index
    ]


    y_train = y.iloc[
        train_index
    ]

    y_valid = y.iloc[
        valid_index
    ]


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = lgb.LGBMClassifier(
        **model_params
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    valid_probability = model.predict_proba(
        X_valid
    )[:, 1]


    # --------------------------------------------------------
    # Use 0.5 as initial classification threshold
    # --------------------------------------------------------

    valid_prediction = (
        valid_probability >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_valid,
        valid_probability
    )


    pr_auc = average_precision_score(
        y_valid,
        valid_probability
    )


    precision = precision_score(
        y_valid,
        valid_prediction,
        zero_division=0
    )


    recall = recall_score(
        y_valid,
        valid_prediction,
        zero_division=0
    )


    f1 = f1_score(
        y_valid,
        valid_prediction,
        zero_division=0
    )


    fold_results.append(
        {
            "fold": fold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "validation_samples": len(
                valid_index
            ),
            "validation_positives": int(
                y_valid.sum()
            ),
            "validation_controls": int(
                (y_valid == 0).sum()
            )
        }
    )


    # --------------------------------------------------------
    # Save predictions for pooled evaluation
    # --------------------------------------------------------

    all_actual.extend(
        y_valid.tolist()
    )

    all_probabilities.extend(
        valid_probability.tolist()
    )

    all_predictions.extend(
        valid_prediction.tolist()
    )


    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )

    print(
        "PR-AUC:",
        round(pr_auc, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )


# ============================================================
# 12. Convert CV results
# ============================================================

cv_results = pd.DataFrame(
    fold_results
)


# ============================================================
# 13. CV summary
# ============================================================

print(
    "\n========================================"
)

print(
    "CROSS-VALIDATION SUMMARY"
)

print(
    "========================================"
)

print(
    cv_results.to_string(
        index=False
    )
)


print(
    "\nMean metrics:"
)

print(
    "ROC-AUC:",
    round(
        cv_results["roc_auc"].mean(),
        4
    )
)

print(
    "PR-AUC:",
    round(
        cv_results["pr_auc"].mean(),
        4
    )
)

print(
    "Precision:",
    round(
        cv_results["precision"].mean(),
        4
    )
)

print(
    "Recall:",
    round(
        cv_results["recall"].mean(),
        4
    )
)

print(
    "F1:",
    round(
        cv_results["f1"].mean(),
        4
    )
)


# ============================================================
# 14. Pooled out-of-fold evaluation
# ============================================================

all_actual = np.array(
    all_actual
)

all_probabilities = np.array(
    all_probabilities
)

all_predictions = np.array(
    all_predictions
)


pooled_roc_auc = roc_auc_score(
    all_actual,
    all_probabilities
)


pooled_pr_auc = average_precision_score(
    all_actual,
    all_probabilities
)


pooled_precision = precision_score(
    all_actual,
    all_predictions,
    zero_division=0
)


pooled_recall = recall_score(
    all_actual,
    all_predictions,
    zero_division=0
)


pooled_f1 = f1_score(
    all_actual,
    all_predictions,
    zero_division=0
)


print(
    "\n========================================"
)

print(
    "POOLED OUT-OF-FOLD METRICS"
)

print(
    "========================================"
)

print(
    "ROC-AUC:",
    round(
        pooled_roc_auc,
        4
    )
)

print(
    "PR-AUC:",
    round(
        pooled_pr_auc,
        4
    )
)

print(
    "Precision:",
    round(
        pooled_precision,
        4
    )
)

print(
    "Recall:",
    round(
        pooled_recall,
        4
    )
)

print(
    "F1:",
    round(
        pooled_f1,
        4
    )
)


# ============================================================
# 15. Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_actual,
    all_predictions
)


print(
    "\nConfusion matrix:"
)

print(
    cm
)


print(
    "\nClassification report:"
)

print(
    classification_report(
        all_actual,
        all_predictions,
        target_names=[
            "control",
            "landslide"
        ],
        zero_division=0
    )
)


# ============================================================
# 16. Train final model on all data
# ============================================================

print(
    "\nTraining final LightGBM model "
    "on all 382 samples..."
)


final_model = lgb.LGBMClassifier(
    **model_params
)


final_model.fit(
    X,
    y
)


# ============================================================
# 17. Feature importance
# ============================================================

importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": final_model.feature_importances_
    }
)


importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(
    drop=True
)


print(
    "\nFeature importance:"
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 18. Save feature importance
# ============================================================

importance.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False
)


# ============================================================
# 19. Save cross-validation results
# ============================================================

cv_results.to_csv(
    CV_RESULTS_PATH,
    index=False
)


# ============================================================
# 20. Save model + metadata
# ============================================================

model_package = {
    "model": final_model,

    "features": FEATURE_COLUMNS,

    "cv_results": cv_results,

    "pooled_metrics": {
        "roc_auc": pooled_roc_auc,
        "pr_auc": pooled_pr_auc,
        "precision": pooled_precision,
        "recall": pooled_recall,
        "f1": pooled_f1
    },

    "scale_pos_weight": scale_pos_weight,

    "random_seed": RANDOM_SEED
}


joblib.dump(
    model_package,
    MODEL_PATH
)


# ============================================================
# 21. Final output
# ============================================================

print(
    "\n========================================"
)

print(
    "LIGHTGBM TRAINING COMPLETED"
)

print(
    "========================================"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Feature importance:",
    FEATURE_IMPORTANCE_PATH
)

print(
    "CV results:",
    CV_RESULTS_PATH
)

print(
    "Features used:",
    len(FEATURE_COLUMNS)
)

print(
    "\nModel package saved successfully."
)

