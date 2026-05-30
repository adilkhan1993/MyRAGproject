from fastapi.testclient import TestClient
from app.main import app

# Мы жестко задаем поддельный IP-адрес прямо в сам тестовый клиент!
client = TestClient(app, client=("192.168.1.100", 50000))

def test_rate_limiting():
    """Проверяем, что при спаме сервер выдает ошибку 429"""
    # Отправляем 10 запросов (это наш разрешенный лимит)
    for _ in range(10):
        res = client.post("/generate", json={"prompt": "Тест лимита"})
        assert res.status_code == 200
    
    # 11-й запрос должен быть заблокирован
    response = client.post("/generate", json={"prompt": "Тест лимита"})
    assert response.status_code == 429