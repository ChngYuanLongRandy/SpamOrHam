from fastapi import FastAPI, APIRouter
from api.api.config import SETTINGS
from api.api.v1.model import ROUTER

APP = FastAPI(title=SETTINGS.API_NAME)
API_ROUTER = APIRouter()
API_ROUTER.include_router(ROUTER, prefix="/logreg", tags=["Logistic Regression"])
APP.include_router(API_ROUTER, prefix=SETTINGS.API_STR)