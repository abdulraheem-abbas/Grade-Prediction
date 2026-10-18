"""Run with: uvicorn student_performance_api:app --reload."""

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pycaret.classification import load_model, predict_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).resolve().with_name(
        "student_performance_model")
    app.state.model = load_model(str(model_path), verbose=False)
    yield


app = FastAPI(title="Student Performance API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse(Path(__file__).resolve().with_name("index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    Age: int = Query(..., description="Student age"),
    StudyTimeWeekly: float = Query(...,
                                   description="StudyTimeWeekly of the student"),
    Absences: int = Query(..., description="Absences of the student"),
    Gender: int = Query(..., description="0: male, 1: female"),
    Ethnicity: int = Query(..., description="Ethnicity of the student"),
    ParentalEducation: int = Query(...,
                                   description="ParentalEducation of the student"),
    Tutoring: int = Query(..., description="Tutoring of the student"),
    ParentalSupport: int = Query(...,
                                 description="ParentalSupport of the student"),
    Extracurricular: int = Query(...,
                                 description="Extracurricular of the student"),
    Sports: int = Query(..., description="Sports of the student"),
    Music: int = Query(..., description="Music of the student"),
    Volunteering: int = Query(..., description="Volunteering of the student"),
):
    data = {
        "Age": Age,
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "Gender": Gender,
        "Ethnicity": Ethnicity,
        "ParentalEducation": ParentalEducation,
        "Tutoring": Tutoring,
        "ParentalSupport": ParentalSupport,
        "Extracurricular": Extracurricular,
        "Sports": Sports,
        "Music": Music,
        "Volunteering": Volunteering,
    }
    df = pd.DataFrame([data])

    predictions = predict_model(app.state.model, data=df, verbose=False)

    predicted_grade = predictions["prediction_label"].iloc[0]
    grade_map = {
        0: '(90 -100) Excellent',
        1: '(80 - below 90%) Very Good',
        2: '(70 - below 80%) Good',
        3: '(60 - below 70%) Pass',
        4: '(Below 60%) Fail'
    }
    grade = grade_map.get(predicted_grade, "Unknown")
    return {"expected grade": grade}
