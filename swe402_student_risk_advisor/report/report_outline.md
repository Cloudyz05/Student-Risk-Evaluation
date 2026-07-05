# SWE402 Report Outline - Student Academic Risk Advisor

## 1. Problem Statement and Project Objectives

This project addresses the problem of identifying students who may be at risk of weak final academic performance. The objective is to build a data mining and machine learning solution that predicts a student's academic risk category, then connect the model to an n8n agentic workflow that can interpret the prediction and recommend an action.

## 2. Dataset Description

Dataset: UCI Student Performance dataset.

Initial project file used: `student-por.csv`, the Portuguese language student performance dataset.

Key target:

- `G3`: final grade from 0 to 20.

Risk label created for this project:

- `High Risk`: `G3 < 10`
- `Medium Risk`: `10 <= G3 <= 13`
- `Low Risk`: `G3 >= 14`

Important input attributes:

- Demographic attributes: school, sex, age, address.
- Family and support attributes: parent education, family support, school support.
- Study behavior attributes: study time, failures, absences.
- Prior performance attributes: `G1` and `G2`.

## 3. Agentic AI Architecture

Required layers:

- Trigger Layer: n8n webhook receives a student profile.
- Data Ingestion Layer: n8n reads and formats the submitted profile.
- Prediction Layer: n8n sends the profile to the FastAPI `/predict` endpoint.
- AI Agent Layer: the AI Agent interprets the risk level and writes a recommendation.
- Action / Output Layer: n8n returns the result, sends a notification, or logs the case.

## 4. Data Preparation and Preprocessing

Planned content:

- Load the CSV file using semicolon separator.
- Check row count, columns, data types, and missing values.
- Create the `risk_label` target from `G3`.
- Split features and target.
- One-hot encode categorical variables.
- Keep numerical variables unchanged.

## 5. Exploratory Data Analysis

Planned figures:

- Risk category distribution.
- `G2` by risk category.
- Absences by risk category.
- Numeric correlation heatmap.

## 6. Predictive Model Development and Evaluation

Initial model:

- Random Forest Classifier.

Reasons:

- Handles mixed feature importance well after one-hot encoding.
- Works well for tabular classification.
- Does not require strict linear assumptions.
- Can handle non-linear relationships between student attributes and risk category.

Metrics:

- Accuracy.
- Weighted F1-score.
- Classification report.
- Confusion matrix.

## 7. Model Deployment and Integration

Deployment approach:

- FastAPI application exposes `/predict`, `/health`, and `/metadata`.
- The trained model is saved using joblib.
- n8n calls the deployed API using an HTTP Request node.

## 8. n8n Agentic Workflow

Main workflow:

1. Webhook trigger receives student data.
2. HTTP Request node calls FastAPI prediction endpoint.
3. AI Agent node receives prediction result and student summary.
4. AI Agent produces a clear support recommendation.
5. Output node returns or sends the recommendation.

## 9. Limitations and Future Improvements

Possible limitations:

- The dataset is from Portuguese schools, so it may not fully represent other education systems.
- The model uses historical data and may reflect patterns specific to the original dataset.
- The first version predicts broad risk categories instead of exact student needs.
- The workflow recommendation should support human decision-making, not replace advisors or lecturers.

Possible improvements:

- Train with a larger and more current institutional dataset.
- Add model comparison with Logistic Regression, Decision Tree, and Gradient Boosting.
- Add explainability using feature importance or SHAP.
- Connect n8n to email, Google Sheets, or a student support ticket system.
