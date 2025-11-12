from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import plot_importance
from xgboost import XGBClassifier
from matplotlib import pyplot
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

model_dir = Path("./model")
baseline_model_path = model_dir / "initial_review_classifier.pkl"
model = joblib.load(baseline_model_path)


df_path = Path(__file__).resolve().parent / "processed-fake-reviews.csv"
df = pd.read_csv(df_path)

X = pd.get_dummies(
    df.drop(columns=["label", "text_", "cleaned_text"]), 
    columns=["category"]
)

label_map = {"OR": 1, "CG": 0}

y = df["label"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# Feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df)
plot_importance(model)
pyplot.show()


# Tune hyperparameters
'''parameters = {
    "n_estimators": [50, 100, 200, 500],
    "learning_rate": [0.1, 0.3, 0.6, 1.0],
    "max_depth": [3, 6, 10],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.1, 0.5, 1.0, 1.5],
}


xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=parameters,
    scoring="roc_auc",
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all CPU core
    verbose=1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_

print("\n" + "=" * 50)
print("GRID SEARCH RESULTS")
print("=" * 50)
print(f"Best parameters: {best_params}")
print(f"Best cross-validation AUC score: {best_score:.4f}")
print("=" * 50)

best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]
best_test_auc = roc_auc_score(y_test, best_prob)

print("\nBest Model Performance on Test Set:")
print("Classification Report:\n", classification_report(y_test, best_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, best_pred))
print(f"Test AUC Score: {best_test_auc:.4f}")


baseline_pred = model.predict(X_test)
baseline_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("BASELINE vs BEST MODEL COMPARISON")
print("=" * 50)
print(f"Baseline AUC: {roc_auc_score(y_test, baseline_prob):.4f}")
print(f"Best Model AUC: {best_test_auc:.4f}")

joblib.dump(best_model, model_dir / "best_params_review_classifier.pkl")

feature_names = X.columns.tolist()

with open(model_dir / "best_params_feature_names.json", "w") as f:
    json.dump(feature_names, f)

best_params_model_metadata = {
    "best_params": best_params,
    "test_auc_score": float(best_test_auc),
    "best_cv_score": float(best_score),
    "num_features": len(feature_names),
    "label_mapping": label_map,
    "training_samples": len(X_train),
    "test_samples": len(X_test)
}

with open(model_dir / "best_params_model_metadata.json", "w") as f:
    json.dump(best_params_model_metadata, f, indent=2)

print(f"\nModel saved successfully in '{model_dir}' directory!")'''



# Feature selection
'''# Only keep features with importance >= thresh
thresh = 0.002

selection = SelectFromModel(model, threshold=thresh, prefit=True)

select_X_train = selection.transform(X_train)

best_params = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "max_depth": 6,
    "reg_alpha": 1.0,
    "reg_lambda": 1.5
}

selection_model = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

selection_model.fit(select_X_train, y_train)

select_X_test = selection.transform(X_test)
selection_model_pred = selection_model.predict(select_X_test)
selection_model_prob = selection_model.predict_proba(select_X_test)[:, 1]
selection_test_auc = roc_auc_score(y_test, selection_model_prob)

print("\nSelected Features Model on Test Set:")
print("Classification Report:\n", classification_report(y_test, selection_model_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, selection_model_pred))
print(f"Test AUC Score: {selection_test_auc:.4f}")

joblib.dump(selection_model, model_dir / f"selection_{thresh}_review_classifier.pkl")

feature_names = X.columns[selection.get_support()].tolist()

with open(model_dir / f"selection_{thresh}_feature_names.json", "w") as f:
    json.dump(feature_names, f)

selection_model_metadata = {
    "best_params": best_params,
    "test_auc_score": float(selection_test_auc),
    "threshold": thresh,
    "num_features": len(feature_names),
    "label_mapping": label_map,
    "training_samples": len(select_X_train),
    "test_samples": len(select_X_test)
}

with open(model_dir / f"selection_{thresh}_model_metadata.json", "w") as f:
    json.dump(selection_model_metadata, f, indent=2)
    
print(f"\nModel saved successfully in '{model_dir}' directory!")'''