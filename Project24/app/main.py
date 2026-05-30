from fastapi import FastAPI, Request
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# Импортируем наши защитные модули из других файлов
from app.models import PromptRequest
from app.rate_limiter import limiter
from app.security import check_for_prompt_injection

app = FastAPI(title="Secure GenAI API")

# Подключаем Rate Limiter к нашему приложению
app.state.limiter = limiter
# Учим приложение правильно выдавать ошибку 429 (Too Many Requests), если лимит превышен
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
async def root():
    return {"message": "Welcome to Secure GenAI API"}


# Создаем эндпоинт и вешаем на него лимит: 10 запросов в минуту для одного IP
@app.post("/generate")
@limiter.limit("10/minute")
async def generate_text(request: Request, body: PromptRequest):
    """
    FastAPI автоматически проверяет данные через PromptRequest. 
    Если данные плохие (например, пустой текст) - он сам выдаст ошибку 422.
    """
    
    # Дополнительно проверяем текст на попытки взлома (Prompt Injection)
    check_for_prompt_injection(body.prompt)
    
    # Если все проверки пройдены успешно, возвращаем ответ
    return {
        "message": "Запрос успешно обработан и прошел все проверки безопасности!",
        "data": {
            "prompt": body.prompt,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature
        }
    }