import os
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from models.disaster_model import get_model, save_model

print("Loading historical road-incident dataset...")
roads_path = "Data/processed/assam_roads_final_labeled.gpkg"

if not os.path.exists(roads_path):
    raise FileNotFoundError(f"Could not find {roads_path}. Please make sure your dataset is generated first.")

df = gpd.read_file(roads_path)

# Ensure CRS matches WGS84
if df.crs != "EPSG:4326":
    df = df.to_crs("EPSG:4326")

# Define features and target
feature_cols = ['mean_elevation', 'max_elevation', 'mean_slope', 'max_slope']
target_col = 'disaster_risk'

# Clean data by dropping missing values
data_subset = df.dropna(subset=feature_cols + [target_col])
X = data_subset[feature_cols]
y = data_subset[target_col]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Random Forest classifier on {len(X_train):,} road segments...")
model = get_model()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print("\n--- Model Performance Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
save_model(model, model_path="models/disaster_risk_model.pkl")