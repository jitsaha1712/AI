
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

MODEL_PATH = (
    r"C:\NERProject\models"
    r"\lgbm_landslide.pkl"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\landslide_threshold_results.csv"
)

THRESHOLD_PATH = (
    r"C:\NERProject\Data\processed"
    r"\landslide_selected_threshold.txt"
)


# ============================================================
# 2. Configuration
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 3. Features
# ============================================================

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
# 4. Read dataset
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
# 5. Prepare X, y and groups
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    "label"
].astype(int)

groups = pd.to_datetime(
    df["event_date"]
).dt.strftime(
    "%Y-%m-%d"
)


# ============================================================
# 6. Class weight
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


# ============================================================
# 7. Same LightGBM configuration as Step 18
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
# 8. Grouped cross-validation
# ============================================================

print(
    "\nGenerating out-of-fold probabilities..."
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_SEED
)


oof_probability = np.zeros(
    len(df),
    dtype=float
)


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
        "Fold",
        fold,
        "/ 5"
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


    oof_probability[
        valid_index
    ] = model.predict_proba(
        X_valid
    )[:, 1]


# ============================================================
# 9. Check pooled probability performance
# ============================================================

roc_auc = roc_auc_score(
    y,
    oof_probability
)

pr_auc = average_precision_score(
    y,
    oof_probability
)


print(
    "\n========================================"
)

print(
    "OUT-OF-FOLD PROBABILITY CHECK"
)

print(
    "========================================"
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


# ============================================================
# 10. Test thresholds
# ============================================================

print(
    "\nTesting thresholds..."
)


thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


results = []


for threshold in thresholds:

    predictions = (
        oof_probability >= threshold
    ).astype(int)


    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )


    accuracy = accuracy_score(
        y,
        predictions
    )


    # --------------------------------------------------------
    # F2 gives more importance to recall than precision.
    #
    # F2 = 5 * precision * recall / (4 * precision + recall)
    # --------------------------------------------------------

    if (
        precision + recall
    ) > 0:

        f2 = (
            5
            * precision
            * recall
            /
            (
                4 * precision
                +
                recall
            )
        )

    else:

        f2 = 0.0


    predicted_landslides = int(
        predictions.sum()
    )


    results.append(
        {
            "threshold": round(
                float(threshold),
                2
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "accuracy": accuracy,
            "predicted_landslides": predicted_landslides
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 11. Print threshold table
# ============================================================

print(
    "\nThreshold results:"
)

print(
    results_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
            "f2": "{:.4f}".format,
            "accuracy": "{:.4f}".format
        }
    )
)


# ============================================================
# 12. Select threshold using F2
# ============================================================
#
# F2 emphasizes recall.
#
# We additionally require precision >= 0.30 so that the
# selected threshold does not produce an excessive number
# of false alarms.
#

eligible = results_df[
    results_df["precision"] >= 0.30
].copy()


if len(eligible) == 0:

    print(
        "\nNo threshold achieved precision >= 0.30."
    )

    # Fall back to maximum F2.
    best_row = results_df.loc[
        results_df["f2"].idxmax()
    ]

else:

    best_row = eligible.loc[
        eligible["f2"].idxmax()
    ]


selected_threshold = float(
    best_row["threshold"]
)


# ============================================================
# 13. Selected threshold metrics
# ============================================================

print(
    "\n========================================"
)

print(
    "SELECTED THRESHOLD"
)

print(
    "========================================"
)

print(
    "Threshold:",
    selected_threshold
)

print(
    "Precision:",
    round(
        float(best_row["precision"]),
        4
    )
)

print(
    "Recall:",
    round(
        float(best_row["recall"]),
        4
    )
)

print(
    "F1:",
    round(
        float(best_row["f1"]),
        4
    )
)

print(
    "F2:",
    round(
        float(best_row["f2"]),
        4
    )
)

print(
    "Accuracy:",
    round(
        float(best_row["accuracy"]),
        4
    )
)

print(
    "Predicted landslides:",
    int(
        best_row["predicted_landslides"]
    )
)


# ============================================================
# 14. Save threshold results
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 15. Save selected threshold
# ============================================================

with open(
    THRESHOLD_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        str(selected_threshold)
    )


# ============================================================
# 16. Save threshold information into model package
# ============================================================

if os.path.exists(
    MODEL_PATH
):

    package = joblib.load(
        MODEL_PATH
    )

    package[
        "selected_threshold"
    ] = selected_threshold

    package[
        "threshold_method"
    ] = (
        "Maximum F2 among thresholds "
        "with precision >= 0.30"
    )

    joblib.dump(
        package,
        MODEL_PATH
    )


# ============================================================
# 17. Final output
# ============================================================

print(
    "\n========================================"
)

print(
    "THRESHOLD TUNING COMPLETED"
)

print(
    "========================================"
)

print(
    "Threshold results:",
    OUTPUT_CSV
)

print(
    "Selected threshold:",
    THRESHOLD_PATH
)

print(
    "Updated model:",
    MODEL_PATH
)

