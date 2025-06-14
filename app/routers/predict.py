from fastapi import APIRouter
from pydantic import BaseModel
from app.models.predict import predict_tf_style

router = APIRouter(
    prefix="/api/v1",
    tags=["predict"],
    responses={404: {"description": "Not found"}},
)

class PredictRequest(BaseModel):
    text: str

@router.post("/predict")
def predict(request: PredictRequest):
    """
    Predict the Thinking (T) vs Feeling (F) style of the input text.
    
    - **text**: The input text to analyze
    """
    result = predict_tf_style(request.text)
    return {"input": request.text, "result": result}
