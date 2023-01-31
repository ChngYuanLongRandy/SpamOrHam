import asyncio
import httpx
import pytest
from asgi_lifespan import LifespanManager
import pytest_asyncio
from api.api.main import APP
from api.api.config import SETTINGS
from fastapi import status


@pytest.fixture(scope="session")
def event_loop():
    """
    Ensures that we will only work within the same event loop instance
    Session scope means that the fixture is created once at the beginning
    of the whole test run
    """
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def test_client():
    """
    Creates an instance of HTTPX for testing API endpoints.
    With LifespanManager, it ensures that startup and shutdown events are
    executed
    With httpx.AsyncClient, it ensures that a HTTP session is ready
    """
    async with LifespanManager(APP):
        async with httpx.AsyncClient(app=APP, base_url="http://app.io") as test_client:
            yield test_client

@pytest.mark.asyncio
async def test_connection(test_client: httpx.AsyncClient):
    """
    Test the connection of the API. As the base endpoint is not defined
    we test the Swagger UI. It should return a 200 if it is connected
    """
    response = await test_client.get("/docs")

    assert response.status_code == status.HTTP_200_OK

@pytest.mark.asyncio
async def test_api_string(test_client: httpx.AsyncClient):
    """
    Test the connection of the API to get API version
    """
    response = await test_client.get(f"{SETTINGS.API_STR}/logreg/version")

    assert response.status_code == status.HTTP_200_OK

@pytest.mark.asyncio
class TestPrediction:
    """
    Runs some test to catch invalid inputs and test the format of the output
    As well as some sanity checks
    """
    async def test_invalid_predict(self, test_client:httpx.AsyncClient):
        payload = "Unable to take a string"
        response = await test_client.post(f"{SETTINGS.API_STR}/logreg/predict", json=payload)

        assert response.status_code != status.HTTP_200_OK

    async def test_invalid_predict_proba(self, test_client:httpx.AsyncClient):
        payload = "Unable to take a string"
        response = await test_client.post(f"{SETTINGS.API_STR}/logreg/predict_proba", json=payload)

        assert response.status_code != status.HTTP_200_OK

    async def test_valid_predict_ham(self, test_client:httpx.AsyncClient):

        test_string = "Eh, want to go out anot"

        payload = {"text": test_string}
        response = await test_client.post(f"{SETTINGS.API_STR}/logreg/predict", json=payload)

        assert response.status_code == status.HTTP_200_OK
        assert int(response.text) == 0

    async def test_valid_predict_proba_ham(self, test_client:httpx.AsyncClient):

        test_string = "Eh, want to go out anot"

        payload = {"text": test_string}
        response = await test_client.post(f"{SETTINGS.API_STR}/logreg/predict_proba", json=payload)

        assert response.status_code == status.HTTP_200_OK

        prediction = response.json()

        assert list(prediction.keys()) == ['ham','spam']
        assert prediction['ham'] > 0.5

    async def test_valid_predict_spam(self, test_client:httpx.AsyncClient):

        test_string = "Call 8080-FREE-STUFF for some free stuff! Call now!"

        payload = {"text": test_string}
        response = await test_client.post(f"{SETTINGS.API_STR}/logreg/predict", json=payload)

        assert response.status_code == status.HTTP_200_OK
        assert int(response.text) == 1

    async def test_valid_predict_proba_spam(self, test_client:httpx.AsyncClient):

        test_string = "Call 8080-FREE-STUFF for some free stuff! Call now!"

        payload = {"text": test_string}
        response = await test_client.post(f"{SETTINGS.API_STR}/logreg/predict_proba", json=payload)

        assert response.status_code == status.HTTP_200_OK

        prediction = response.json()

        assert list(prediction.keys()) == ['ham','spam']
        assert prediction['spam'] > 0.5