# app/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from model.predict import predict_tf_style

router = APIRouter()


class PredictRequest(BaseModel):
    text: str


@router.post("/predict")
def predict(request: PredictRequest):
    result = predict_tf_style(request.text)
    return {"input": request.text, "result": result}
