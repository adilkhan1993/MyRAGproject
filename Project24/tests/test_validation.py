from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_successful_request():
    """Проверяем, что нормальный запрос проходит (код 200)"""
    response = client.post("/generate", json={"prompt": "Привет, мир!"})
    assert response.status_code == 200

def test_empty_prompt():
    """Проверяем, что пустой запрос блокируется (код 422)"""
    response = client.post("/generate", json={"prompt": "   "})
    assert response.status_code == 422

def test_prompt_too_long():
    """Проверяем, что слишком длинный текст блокируется (код 422)"""
    long_prompt = "A" * 2001
    response = client.post("/generate", json={"prompt": long_prompt})
    assert response.status_code == 422

def test_temperature_out_of_bounds():
    """Проверяем, что неправильная температура блокируется (код 422)"""
    response = client.post("/generate", json={"prompt": "Тест", "temperature": 5.0})
    assert response.status_code == 422