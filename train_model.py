import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from pathlib import Path
import joblib
import json

df_path = Path(__file__).resolve().parent / "processed-fake-reviews.csv"
df = pd.read_csv(df_path)

print(f"Dataset loaded with {len(df)} reviews")
print(f"Columns in dataset: {list(df.columns)}")

# Prepare features (X) by removing text columns and the label column
X = pd.get_dummies(
    df.drop(columns=["label", "text_", "cleaned_text"]), 
    columns=["category"]
)

# Convert labels to numbers: OR (Original/real Reviews) = 1, CG (Computer-Generated) = 0
label_map = {"OR": 1, "CG": 0}

y = df["label"].map(label_map)

print(f"Features shape: {X.shape}")
print(f"Labels distribution:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Create XGBoost classifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

print("Training the model...")


# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print AUC score
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nAUC Score: {auc_score:.4f}")


# Create directory for saving the model
model_dir = Path("./model")
model_dir.mkdir(parents=True, exist_ok=True)

# Save the trained model
joblib.dump(model, model_dir / "initial_review_classifier.pkl")

# Save feature names for future use
feature_names = X.columns.tolist()

with open(model_dir / "initial_feature_names.json", "w") as f:
    json.dump(feature_names, f)

# Save model metadata
model_metadata = {
    "test_auc_score": float(auc_score),
    "num_features": len(feature_names),
    "label_mapping": label_map,
    "training_samples": len(X_train),
    "test_samples": len(X_test)
}

with open(model_dir / "initial_model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

print(f"\nModel saved successfully in '{model_dir}' directory!")