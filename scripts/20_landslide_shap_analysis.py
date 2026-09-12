
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


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

OUTPUT_DIR = (
    r"C:\NERProject\Data\processed"
    r"\shap_landslide"
)

GLOBAL_IMPORTANCE_CSV = os.path.join(
    OUTPUT_DIR,
    "shap_global_importance.csv"
)

SAMPLE_EXPLANATION_CSV = os.path.join(
    OUTPUT_DIR,
    "shap_sample_explanation.csv"
)

SHAP_SUMMARY_PLOT = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)

SHAP_BAR_PLOT = os.path.join(
    OUTPUT_DIR,
    "shap_bar.png"
)


# ============================================================
# 2. Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. Load dataset
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
# 4. Load saved model
# ============================================================

print("\nLoading LightGBM model...")

package = joblib.load(
    MODEL_PATH
)


model = package["model"]

FEATURE_COLUMNS = package["features"]

selected_threshold = package.get(
    "selected_threshold",
    0.20
)


print(
    "Model loaded successfully."
)

print(
    "Selected threshold:",
    selected_threshold
)

print(
    "Features:",
    len(FEATURE_COLUMNS)
)


# ============================================================
# 5. Prepare feature matrix
# ============================================================

X = df[
    FEATURE_COLUMNS
].copy()


# ============================================================
# 6. Create SHAP explainer
# ============================================================

print(
    "\nCreating SHAP explainer..."
)

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# 7. Calculate SHAP values
# ============================================================

print(
    "Calculating SHAP values..."
)

shap_values = explainer.shap_values(
    X
)


# ============================================================
# 8. Handle SHAP output format
# ============================================================
#
# Depending on SHAP version, binary LightGBM models can return:
#
# 1. A 2D array
# 2. A list containing arrays
#

if isinstance(
    shap_values,
    list
):

    if len(shap_values) == 2:

        shap_values = shap_values[1]

    else:

        shap_values = shap_values[0]


shap_values = np.asarray(
    shap_values
)


print(
    "SHAP matrix shape:",
    shap_values.shape
)


# ============================================================
# 9. Global SHAP importance
# ============================================================

mean_abs_shap = np.mean(
    np.abs(
        shap_values
    ),
    axis=0
)


global_importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "mean_abs_shap": mean_abs_shap
    }
)


global_importance = (
    global_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\n========================================"
)

print(
    "GLOBAL SHAP IMPORTANCE"
)

print(
    "========================================"
)

print(
    global_importance.to_string(
        index=False
    )
)


# ============================================================
# 10. Save global importance
# ============================================================

global_importance.to_csv(
    GLOBAL_IMPORTANCE_CSV,
    index=False
)


# ============================================================
# 11. SHAP summary plot
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    SHAP_SUMMARY_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. SHAP bar plot
# ============================================================

print(
    "Creating SHAP bar plot..."
)

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    SHAP_BAR_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. Individual sample explanation
# ============================================================
#
# Choose the known landslide with the highest predicted
# probability.
#

probabilities = model.predict_proba(
    X
)[:, 1]


df_with_probability = df.copy()

df_with_probability[
    "p_landslide"
] = probabilities


landslide_rows = df_with_probability[
    df_with_probability["label"] == 1
]


highest_index = (
    landslide_rows["p_landslide"]
    .idxmax()
)


sample_position = (
    df.index.get_loc(
        highest_index
    )
)


sample_probability = probabilities[
    sample_position
]


sample_shap = shap_values[
    sample_position
]


sample_explanation = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "feature_value": X.iloc[
            sample_position
        ].values,
        "shap_value": sample_shap
    }
)


sample_explanation[
    "absolute_shap"
] = np.abs(
    sample_explanation[
        "shap_value"
    ]
)


sample_explanation = (
    sample_explanation
    .sort_values(
        "absolute_shap",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. Save individual explanation
# ============================================================

sample_explanation.to_csv(
    SAMPLE_EXPLANATION_CSV,
    index=False
)


# ============================================================
# 15. Print individual explanation
# ============================================================

print(
    "\n========================================"
)

print(
    "HIGHEST-PROBABILITY HISTORICAL LANDSLIDE"
)

print(
    "========================================"
)

print(
    "Sample ID:",
    df.iloc[
        sample_position
    ]["sample_id"]
)

print(
    "Original ID:",
    df.iloc[
        sample_position
    ]["original_id"]
)

print(
    "Date:",
    df.iloc[
        sample_position
    ]["event_date"]
)

print(
    "Location:",
    df.iloc[
        sample_position
    ]["longitude"],
    ",",
    df.iloc[
        sample_position
    ]["latitude"]
)

print(
    "Actual label:",
    df.iloc[
        sample_position
    ]["label"]
)

print(
    "Predicted P(landslide):",
    round(
        float(sample_probability),
        4
    )
)

print(
    "Selected threshold:",
    selected_threshold
)


print(
    "\nMain SHAP contributors:"
)

print(
    sample_explanation[
        [
            "feature",
            "feature_value",
            "shap_value"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 16. Model-level explanation
# ============================================================

print(
    "\n========================================"
)

print(
    "MODEL EXPLANATION COMPLETED"
)

print(
    "========================================"
)

print(
    "Global importance:",
    GLOBAL_IMPORTANCE_CSV
)

print(
    "Sample explanation:",
    SAMPLE_EXPLANATION_CSV
)

print(
    "Summary plot:",
    SHAP_SUMMARY_PLOT
)

print(
    "Bar plot:",
    SHAP_BAR_PLOT
)

