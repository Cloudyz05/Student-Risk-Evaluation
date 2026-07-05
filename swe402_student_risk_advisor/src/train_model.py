from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


matplotlib.use("Agg")

import matplotlib.pyplot as plt

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_performance" / "student" / "student-por.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_por_prepared.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_risk_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "student_risk_metadata.json"
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"


RISK_ORDER = ["High Risk", "Medium Risk", "Low Risk"]


def assign_risk_label(final_grade: int) -> str:
    if final_grade < 10:
        return "High Risk"
    if final_grade <= 13:
        return "Medium Risk"
    return "Low Risk"


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df["risk_label"] = df["G3"].apply(assign_risk_label)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def create_figures(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="risk_label", order=RISK_ORDER, hue="risk_label", palette="Set2", legend=False)
    plt.title("Student Risk Category Distribution")
    plt.xlabel("Risk category")
    plt.ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "risk_category_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="risk_label", y="G2", order=RISK_ORDER, hue="risk_label", palette="Set2", legend=False)
    plt.title("Second Period Grade by Risk Category")
    plt.xlabel("Risk category")
    plt.ylabel("G2 grade")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "g2_by_risk_category.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="risk_label", y="absences", order=RISK_ORDER, hue="risk_label", palette="Set2", legend=False)
    plt.title("Absences by Risk Category")
    plt.xlabel("Risk category")
    plt.ylabel("Absences")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "absences_by_risk_category.png", dpi=160)
    plt.close()

    numeric_cols = ["age", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap for Numeric Attributes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "numeric_correlation_heatmap.png", dpi=160)
    plt.close()


def build_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    target = "risk_label"
    drop_columns = ["G3", target]
    X = df.drop(columns=drop_columns)
    y = df[target]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

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

    labels = RISK_ORDER
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics = {
        "dataset": "UCI Student Performance - Portuguese language dataset",
        "row_count": int(df.shape[0]),
        "feature_count": int(X.shape[1]),
        "target_definition": {
            "High Risk": "G3 < 10",
            "Medium Risk": "10 <= G3 <= 13",
            "Low Risk": "G3 >= 14",
        },
        "class_distribution": df[target].value_counts().reindex(labels).fillna(0).astype(int).to_dict(),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
        "features": {
            "categorical": categorical_features,
            "numeric": numeric_features,
        },
    }

    return pipeline, metrics


def save_artifacts(pipeline: Pipeline, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    df = load_and_prepare_data()
    create_figures(df)
    pipeline, metrics = build_model(df)
    save_artifacts(pipeline, metrics)

    print("Training complete.")
    print(f"Prepared data: {PROCESSED_DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Weighted F1: {metrics['weighted_f1']}")
    print("Class distribution:")
    for label, count in metrics["class_distribution"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
