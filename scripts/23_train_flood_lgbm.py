
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
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. Paths
# ============================================================

DATASET_CSV = (
    r"C:\NERProject\Data\processed"
    r"\flood_ml_dataset_2023.csv"
)

MODEL_DIR = (
    r"C:\NERProject\models"
)

MODEL_PATH = (
    r"C:\NERProject\models"
    r"\lgbm_flood.pkl"
)

CV_RESULTS_PATH = (
    r"C:\NERProject\Data\processed"
    r"\flood_model_cv_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    r"C:\NERProject\Data\processed"
    r"\flood_feature_importance.csv"
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

FEATURE_COLUMNS = [
    "rain_1d_2023",
    "rain_3d_2023",
    "rain_7d_2023",
    "max_rain_3d_2023",
    "max_rain_7d_2023",
    "elevation_m",
    "slope_deg",
    "distance_to_river_m"
]


# ============================================================
# 5. Read dataset
# ============================================================

print("Reading flood ML dataset...")

df = pd.read_csv(
    DATASET_CSV
)

print(
    "Samples:",
    len(df)
)


# ============================================================
# 6. Validate columns
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    +
    [
        "edge_id",
        "flood_label"
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


# ============================================================
# 7. Validate feature values
# ============================================================

if df[
    FEATURE_COLUMNS
].isna().any().any():

    raise ValueError(
        "Missing feature values found."
    )


if not np.isfinite(
    df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Invalid numeric feature values found."
    )


# ============================================================
# 8. Prepare X and y
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    "flood_label"
].astype(int)


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
    "Flooded:",
    positive_count
)

print(
    "Non-flooded:",
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

    "n_estimators": 400,

    "learning_rate": 0.03,

    "num_leaves": 31,

    "max_depth": 6,

    "min_child_samples": 30,

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
# 11. 5-fold stratified cross-validation
# ============================================================

print(
    "\nStarting 5-fold cross-validation..."
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)


oof_probabilities = np.zeros(
    len(df),
    dtype=float
)


fold_results = []


for fold, (
    train_index,
    valid_index
) in enumerate(
    cv.split(
        X,
        y
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


    model = lgb.LGBMClassifier(
        **model_params
    )


    model.fit(
        X_train,
        y_train
    )


    probability = model.predict_proba(
        X_valid
    )[:, 1]


    prediction = (
        probability >= 0.5
    ).astype(int)


    oof_probabilities[
        valid_index
    ] = probability


    roc_auc = roc_auc_score(
        y_valid,
        probability
    )


    pr_auc = average_precision_score(
        y_valid,
        probability
    )


    precision = precision_score(
        y_valid,
        prediction,
        zero_division=0
    )


    recall = recall_score(
        y_valid,
        prediction,
        zero_division=0
    )


    f1 = f1_score(
        y_valid,
        prediction,
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
            )
        }
    )


    print(
        "ROC-AUC:",
        round(
            roc_auc,
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            pr_auc,
            4
        )
    )

    print(
        "Precision:",
        round(
            precision,
            4
        )
    )

    print(
        "Recall:",
        round(
            recall,
            4
        )
    )

    print(
        "F1:",
        round(
            f1,
            4
        )
    )


# ============================================================
# 12. CV results
# ============================================================

cv_results = pd.DataFrame(
    fold_results
)


# ============================================================
# 13. Pooled out-of-fold metrics
# ============================================================

oof_predictions = (
    oof_probabilities >= 0.5
).astype(int)


pooled_roc_auc = roc_auc_score(
    y,
    oof_probabilities
)


pooled_pr_auc = average_precision_score(
    y,
    oof_probabilities
)


pooled_precision = precision_score(
    y,
    oof_predictions,
    zero_division=0
)


pooled_recall = recall_score(
    y,
    oof_predictions,
    zero_division=0
)


pooled_f1 = f1_score(
    y,
    oof_predictions,
    zero_division=0
)


# ============================================================
# 14. Print CV summary
# ============================================================

print(
    "\n========================================"
)

print(
    "FLOOD CROSS-VALIDATION SUMMARY"
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
        cv_results[
            "roc_auc"
        ].mean(),
        4
    )
)

print(
    "PR-AUC:",
    round(
        cv_results[
            "pr_auc"
        ].mean(),
        4
    )
)

print(
    "Precision:",
    round(
        cv_results[
            "precision"
        ].mean(),
        4
    )
)

print(
    "Recall:",
    round(
        cv_results[
            "recall"
        ].mean(),
        4
    )
)

print(
    "F1:",
    round(
        cv_results[
            "f1"
        ].mean(),
        4
    )
)


# ============================================================
# 15. Pooled metrics
# ============================================================

print(
    "\n========================================"
)

print(
    "FLOOD POOLED OUT-OF-FOLD METRICS"
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
# 16. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y,
    oof_predictions
)


print(
    "\nConfusion matrix:"
)

print(
    cm
)


# ============================================================
# 17. Classification report
# ============================================================

print(
    "\nClassification report:"
)

print(
    classification_report(
        y,
        oof_predictions,
        target_names=[
            "non-flooded",
            "flooded"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. Train final model on all road segments
# ============================================================

print(
    "\nTraining final flood LightGBM model..."
)

final_model = lgb.LGBMClassifier(
    **model_params
)


final_model.fit(
    X,
    y
)


# ============================================================
# 19. Feature importance
# ============================================================

importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance": (
            final_model
            .feature_importances_
        )
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
# 20. Save feature importance
# ============================================================

importance.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False
)


# ============================================================
# 21. Save CV results
# ============================================================

cv_results.to_csv(
    CV_RESULTS_PATH,
    index=False
)


# ============================================================
# 22. Save model package
# ============================================================

model_package = {
    "model": final_model,

    "features": FEATURE_COLUMNS,

    "flood_label_threshold": 0.10,

    "cv_results": cv_results,

    "pooled_metrics": {
        "roc_auc": pooled_roc_auc,
        "pr_auc": pooled_pr_auc,
        "precision": pooled_precision,
        "recall": pooled_recall,
        "f1": pooled_f1
    },

    "random_seed": RANDOM_SEED
}


joblib.dump(
    model_package,
    MODEL_PATH
)


# ============================================================
# 23. Final output
# ============================================================

print(
    "\n========================================"
)

print(
    "FLOOD LIGHTGBM TRAINING COMPLETED"
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
    "\nFlood model package saved successfully."
)

