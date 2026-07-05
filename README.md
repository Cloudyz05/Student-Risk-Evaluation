# SWE402 Student Academic Risk Advisor

This project builds an agentic AI workflow for SWE402 Data Mining.

Theme: Student Academic Risk Advisor

Goal: predict a student's final academic risk level from the UCI Student Performance dataset, then use an n8n AI Agent workflow to interpret the prediction and recommend an action.

## Main Deliverables

- `notebooks/student_risk_advisor.ipynb` - JupyterLab notebook for data preparation, EDA, model training, and evaluation.
- `src/train_model.py` - reproducible training script.
- `api/main.py` - FastAPI prediction endpoint for n8n integration.
- `n8n/` - workflow planning notes and export draft.
- `report/` - report outline, figures, and writing material.

## Local Setup

Use the project virtual environment:

```powershell
.\.venv\Scripts\activate
```

Train the model:

```powershell
python src\train_model.py
```

Run the API:

```powershell
uvicorn api.main:app --reload
```

Test the API:

```powershell
python src\test_api_payload.py
```

## Project Decision

The model predicts three risk categories from final grade `G3`:

- `High Risk`: `G3 < 10`
- `Medium Risk`: `10 <= G3 <= 13`
- `Low Risk`: `G3 >= 14`

The features include `G1` and `G2` because they are first and second period grades. This makes the use case practical as an early warning tool after the second assessment period, before the final grade is known.
