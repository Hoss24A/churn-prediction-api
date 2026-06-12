import os
import numpy as np
import shap
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model        = joblib.load(os.path.join(BASE_DIR, 'models', 'churn_model.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_names.pkl'))
explainer    = shap.TreeExplainer(model)

def predict_churn(input_dict: dict) -> dict:
    import pandas as pd
    df_input = pd.DataFrame([input_dict])
    X_proc   = preprocessor.transform(df_input)

    proba        = model.predict_proba(X_proc)[0][1]
    prediction   = int(proba >= 0.5)
    shap_vals    = explainer.shap_values(X_proc)[0]

    # Top 5 factors driving this prediction
    top_idx      = np.argsort(np.abs(shap_vals))[::-1][:5]
    top_features = [
        {"feature": feature_names[i], "shap_value": round(float(shap_vals[i]), 4)}
        for i in top_idx
    ]

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": "Churn" if prediction else "No Churn",
        "top_factors": top_features
    }