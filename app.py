import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

model = None

@asynccontextmanager
def lifespan(app: FastAPI):
    global model
    model = pipeline(
        "text-classification",
        model="bhadresh-savani/albert-base-v2-emotion",
    )
    yield

app = FastAPI(
    lifespan=lifespan, docs_url="/", root_path=os.getenv("TFY_SERVICE_ROOT_PATH")
)

class PredictRequest(BaseModel):
    inputs: Union[List[str], str]   # Modify input field name to inputs
    parameters: Dict[str, Any] = Field(default_factory=dict)

EXAMPLES = {
    "example-request": {
        "summary": "Example predict request",
        "value": PredictRequest(inputs=["I am happy", "I am angry", "I am sad"]).dict(),
    }
}

@app.post("/predict")
def predict(request: PredictRequest = Body(..., openapi_examples=EXAMPLES)):
    assert model is not None
    results = model(request.inputs, **request.parameters)  # Note input changed to inputs
    return results
