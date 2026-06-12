# Churn Prediction API

End-to-end ML system that predicts customer churn with XGBoost + SHAP explanations,
served via a FastAPI REST API and deployed on Render.

## Results
| Metric    | Score  |
|-----------|--------|
| ROC-AUC   | 0.856  |
| PR-AUC    | 0.683  |
| Precision | 0.71   |
| Recall    | 0.78   |

## Architecture
[Data] → [Preprocessing] → [SMOTE] → [XGBoost] → [SHAP] → [FastAPI] → [Streamlit]

## Live Demo
API Docs: https://churn-api-xxxx.onrender.com/docs

## Local Setup
\`\`\`bash
git clone ...
pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload
\`\`\`

## Tech Stack
Python · XGBoost · SHAP · FastAPI · Docker · Streamlit · Render
