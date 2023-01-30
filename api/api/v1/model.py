from fastapi import APIRouter
import numpy as np
import pandas as pd
from typing import Dict, List
from api.api.config import SETTINGS
from api.api.schema import Text
from api.api.resources import PRED_MODEL, PIPELINE

ROUTER = APIRouter()

Prediction = int
Prediction_proba = Dict[str,float]
APIVersion = Dict[str, str]

@ROUTER.get("/version")
def get_api_version() -> APIVersion:
    """
    GET method for API version
    """
    return {"API_Version": SETTINGS.API_STR}

@ROUTER.post('/predict')
def predict(text: Text) -> Prediction:
    """Predict Spam or Ham given a string and return the prediction class

    Returns:
        Prediction: 0 for geniune (Ham) 1 for Spam
    """
    text_dict = text.dict()
    input_pd = pd.DataFrame({'spam':1, 'text':text_dict['text']}, index=[0]) 
    X,_ = PIPELINE.transform(input_pd)
    return PRED_MODEL.predict(X)


@ROUTER.post('/predict_proba')
def predict(text: Text):
    """Predict Spam or Ham given a string and return the probability

    Returns:
        Prediction Logit: [Probability for Ham , Probability for Spam]
    """
    text_dict = text.dict()
    input_pd = pd.DataFrame({'spam':1, 'text':text_dict['text']}, index=[0]) 
    X,_ = PIPELINE.transform(input_pd)
    prediction = PRED_MODEL.predict_proba(X)[0]
    return {'ham':prediction[0],'spam':prediction[1]}