import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def classify_flood_risk(row):
    """Classify flood risk based on adjusted thresholds for rainfall, humidity, and temperature."""
    if row["Rainfall"] > 60 and row["Humidity"] > 80 and 20 <= row["Temperature"] <= 35:
        return 2  # High risk (Updated threshold)
    elif row["Rainfall"] > 30 and row["Humidity"] > 60 and 15 <= row["Temperature"] <= 35:
        return 1  # Moderate risk (Same threshold)
    else:
        return 0  # Low risk

# Load dataset
file_path = "flood_dataset.csv"  # Ensure you're using the balanced dataset

df = pd.read_csv(file_path)

# Encode city names
city_encoder = LabelEncoder()
df["Encoded City"] = city_encoder.fit_transform(df["City"])

# Apply flood risk classification
df["Flood Risk Level"] = df.apply(classify_flood_risk, axis=1)

# Select features and target
X = df[["Encoded City", "Temperature", "Humidity", "Rainfall"]]
y = df["Flood Risk Level"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model and encoder
joblib.dump(model, "flood_model.pkl")
joblib.dump(city_encoder, "city_encoder.pkl")
