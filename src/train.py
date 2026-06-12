import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from preprocess import load_and_clean, get_feature_target, build_preprocessor
from sklearn.model_selection import RandomizedSearchCV

# Load data
df = load_and_clean('../data/telco_churn.csv')
X, y = get_feature_target(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess
preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(ohe_features)

# SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_distributions=param_grid,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5),
    scoring='average_precision',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train_res, y_train_res)
print(search.best_params_)

# Evaluate
y_pred  = model.predict(X_test_proc)
y_proba = model.predict_proba(X_test_proc)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:   {average_precision_score(y_test, y_proba):.4f}")
# Target: ROC-AUC > 0.84, PR-AUC > 0.65

# Save artifacts
joblib.dump(model, '../models/churn_model.pkl')
joblib.dump(preprocessor, '../models/preprocessor.pkl')
joblib.dump(feature_names, '../models/feature_names.pkl')
print("Model saved.")