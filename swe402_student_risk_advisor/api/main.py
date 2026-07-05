from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "student_risk_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "student_risk_metadata.json"


app = FastAPI(
    title="Student Academic Risk Advisor API",
    description="Predicts student academic risk for the SWE402 n8n agentic AI workflow.",
    version="1.0.0",
)


class StudentInput(BaseModel):
    school: Literal["GP", "MS"] = "GP"
    sex: Literal["F", "M"] = "F"
    age: int = Field(17, ge=15, le=22)
    address: Literal["U", "R"] = "U"
    famsize: Literal["LE3", "GT3"] = "GT3"
    Pstatus: Literal["T", "A"] = "T"
    Medu: int = Field(2, ge=0, le=4)
    Fedu: int = Field(2, ge=0, le=4)
    Mjob: Literal["teacher", "health", "services", "at_home", "other"] = "other"
    Fjob: Literal["teacher", "health", "services", "at_home", "other"] = "other"
    reason: Literal["home", "reputation", "course", "other"] = "course"
    guardian: Literal["mother", "father", "other"] = "mother"
    traveltime: int = Field(1, ge=1, le=4)
    studytime: int = Field(2, ge=1, le=4)
    failures: int = Field(0, ge=0, le=4)
    schoolsup: Literal["yes", "no"] = "no"
    famsup: Literal["yes", "no"] = "yes"
    paid: Literal["yes", "no"] = "no"
    activities: Literal["yes", "no"] = "yes"
    nursery: Literal["yes", "no"] = "yes"
    higher: Literal["yes", "no"] = "yes"
    internet: Literal["yes", "no"] = "yes"
    romantic: Literal["yes", "no"] = "no"
    famrel: int = Field(4, ge=1, le=5)
    freetime: int = Field(3, ge=1, le=5)
    goout: int = Field(3, ge=1, le=5)
    Dalc: int = Field(1, ge=1, le=5)
    Walc: int = Field(1, ge=1, le=5)
    health: int = Field(3, ge=1, le=5)
    absences: int = Field(2, ge=0)
    G1: int = Field(10, ge=0, le=20)
    G2: int = Field(10, ge=0, le=20)


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Run python src/train_model.py first.")
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@app.get("/")
def root() -> dict:
    return {
        "message": "Student Academic Risk Advisor API is running.",
        "predict_endpoint": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "metadata_exists": METADATA_PATH.exists(),
    }


@app.get("/metadata")
def metadata() -> dict:
    return load_metadata()


@app.post("/predict")
def predict(payload: StudentInput) -> dict:
    model = load_model()
    input_df = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(input_df)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        class_labels = model.classes_
        probability_values = model.predict_proba(input_df)[0]
        probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(class_labels, probability_values)
        }

    recommended_action = {
        "High Risk": "Escalate to academic advisor and recommend immediate support plan.",
        "Medium Risk": "Send study guidance and monitor the student in the next assessment period.",
        "Low Risk": "Send positive feedback and continue normal monitoring.",
    }.get(prediction, "Review the student manually.")

    return {
        "risk_level": prediction,
        "probabilities": probabilities,
        "recommended_action": recommended_action,
        "model_input_summary": {
            "G1": payload.G1,
            "G2": payload.G2,
            "failures": payload.failures,
            "studytime": payload.studytime,
            "absences": payload.absences,
        },
    }
