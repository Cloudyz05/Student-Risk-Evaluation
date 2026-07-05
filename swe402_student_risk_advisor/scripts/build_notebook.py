from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "student_risk_advisor.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # SWE402 Student Academic Risk Advisor

        This notebook prepares the UCI Student Performance dataset, explores important patterns, trains a predictive model, and prepares the model for integration with an n8n agentic workflow.

        Project theme: Student Academic Risk Advisor.

        Main idea: predict whether a student is high, medium, or low academic risk, then allow an n8n AI Agent to interpret the prediction and recommend a support action.
        """
    ),
    markdown(
        """
        ## 1. Import Libraries and Set Paths

        The project uses pandas for data handling, seaborn/matplotlib for visualisation, and scikit-learn for preprocessing, training, and evaluation.
        """
    ),
    code(
        """
        from pathlib import Path
        import json

        import joblib
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_performance" / "student" / "student-por.csv"
        PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_por_prepared.csv"
        MODEL_PATH = PROJECT_ROOT / "models" / "student_risk_model.joblib"
        METADATA_PATH = PROJECT_ROOT / "models" / "student_risk_metadata.json"
        FIGURE_DIR = PROJECT_ROOT / "report" / "figures"

        sns.set_theme(style="whitegrid")
        """
    ),
    markdown(
        """
        ## 2. Load Dataset

        The CSV file uses semicolons as separators, so the `sep=";"` setting is required when loading it.
        """
    ),
    code(
        """
        df = pd.read_csv(RAW_DATA_PATH, sep=";")
        print("Rows and columns:", df.shape)
        df.head()
        """
    ),
    markdown(
        """
        ## 3. Dataset Structure and Quality Check

        This step checks the column types, missing values, duplicate rows, and basic descriptive statistics before modelling.
        """
    ),
    code(
        """
        display(df.info())
        display(df.describe())
        display(df.isna().sum().to_frame("missing_values"))
        print("Duplicate rows:", df.duplicated().sum())
        """
    ),
    markdown(
        """
        ## 4. Create Risk Label

        The original target column is `G3`, the final grade from 0 to 20. For this project, the grade is converted into a three-class risk label.

        - High Risk: G3 < 10
        - Medium Risk: 10 <= G3 <= 13
        - Low Risk: G3 >= 14
        """
    ),
    code(
        """
        def assign_risk_label(final_grade):
            if final_grade < 10:
                return "High Risk"
            if final_grade <= 13:
                return "Medium Risk"
            return "Low Risk"

        risk_order = ["High Risk", "Medium Risk", "Low Risk"]
        df["risk_label"] = df["G3"].apply(assign_risk_label)
        df["risk_label"].value_counts().reindex(risk_order)
        """
    ),
    markdown(
        """
        ## 5. Exploratory Data Analysis

        The visualisations below show how the risk groups differ by prior grade, absences, and other numeric attributes.
        """
    ),
    code(
        """
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="risk_label", order=risk_order, hue="risk_label", palette="Set2", legend=False)
        plt.title("Student Risk Category Distribution")
        plt.xlabel("Risk category")
        plt.ylabel("Number of students")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "risk_category_distribution.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="risk_label", y="G2", order=risk_order, hue="risk_label", palette="Set2", legend=False)
        plt.title("Second Period Grade by Risk Category")
        plt.xlabel("Risk category")
        plt.ylabel("G2 grade")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "g2_by_risk_category.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="risk_label", y="absences", order=risk_order, hue="risk_label", palette="Set2", legend=False)
        plt.title("Absences by Risk Category")
        plt.xlabel("Risk category")
        plt.ylabel("Absences")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "absences_by_risk_category.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        numeric_cols = ["age", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, square=True)
        plt.title("Correlation Heatmap for Numeric Attributes")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "numeric_correlation_heatmap.png", dpi=160)
        plt.show()
        """
    ),
    markdown(
        """
        ## 6. Prepare Features and Target

        The model predicts `risk_label`. The final grade `G3` is removed from the input features because it is used to create the target label.

        `G1` and `G2` are kept because they are earlier period grades. This supports a practical early warning use case after the second assessment period.
        """
    ),
    code(
        """
        target = "risk_label"
        X = df.drop(columns=["G3", target])
        y = df[target]

        categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
        numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

        print("Categorical features:", categorical_features)
        print("Numeric features:", numeric_features)
        """
    ),
    markdown(
        """
        ## 7. Train and Evaluate Model

        A Random Forest Classifier is used because it works well for tabular classification and can model non-linear relationships between student characteristics and risk level.
        """
    ),
    code(
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("numeric", "passthrough", numeric_features),
            ]
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            max_depth=8,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
        print("Weighted F1:", round(f1_score(y_test, predictions, average="weighted"), 4))
        print(classification_report(y_test, predictions, labels=risk_order, zero_division=0))
        """
    ),
    code(
        """
        cm = confusion_matrix(y_test, predictions, labels=risk_order)
        cm_df = pd.DataFrame(cm, index=risk_order, columns=risk_order)

        plt.figure(figsize=(7, 5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 8. Save Model and Metadata

        The saved `joblib` model is used by the FastAPI endpoint. The metadata file stores performance results and feature information for the report.
        """
    ),
    code(
        """
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(PROCESSED_DATA_PATH, index=False)
        joblib.dump(pipeline, MODEL_PATH)

        metrics = {
            "dataset": "UCI Student Performance - Portuguese language dataset",
            "row_count": int(df.shape[0]),
            "feature_count": int(X.shape[1]),
            "target_definition": {
                "High Risk": "G3 < 10",
                "Medium Risk": "10 <= G3 <= 13",
                "Low Risk": "G3 >= 14",
            },
            "class_distribution": df[target].value_counts().reindex(risk_order).fillna(0).astype(int).to_dict(),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
            "features": {
                "categorical": categorical_features,
                "numeric": numeric_features,
            },
        }

        METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics
        """
    ),
    markdown(
        """
        ## 9. API and n8n Integration Plan

        The trained model is served by `api/main.py` using FastAPI.

        Important endpoints:

        - `GET /health`: checks whether the API and model artifacts are available.
        - `GET /metadata`: returns model metadata.
        - `POST /predict`: receives student data and returns risk level, probabilities, and a recommended action.

        In n8n, the workflow should use a Webhook Trigger, HTTP Request node, AI Agent node, and an output/notification node.
        """
    ),
]


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
