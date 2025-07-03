import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Define request model
class PredictRequest(BaseModel):
    inputs: List[str]  # Changed 'input' to 'inputs'
    parameters: Optional[Dict[str, Any]] = None

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Assuming model is already defined
        results = model(request.inputs, **request.parameters)  # Updated to use 'inputs' instead of 'input'
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
