from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# Импортируем наши защитные модули из других файлов
from app.models import PromptRequest
from app.rate_limiter import limiter
from app.security import check_for_prompt_injection

app = FastAPI(title="Secure GenAI API")

# ==========================================
# ДОБАВЛЕНО ДЛЯ ПРОЕКТА 25: Настройка CORS
# Разрешаем браузеру отправлять запросы с нашего будущего фронтенда
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/")
async def root():
    return {"message": "Welcome to Secure GenAI API"}


@app.post("/generate")
@limiter.limit("10/minute")
async def generate_text(request: Request, body: PromptRequest):
    check_for_prompt_injection(body.prompt)
    
    return {
        "message": "Запрос успешно обработан и прошел все проверки безопасности!",
        "data": {
            "prompt": body.prompt,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature
        }
    }


# ==========================================
# ДОБАВЛЕНО ДЛЯ ПРОЕКТА 25: Стриминг
# Эндпоинт, который отдает текст по кусочкам
# ==========================================
@app.post("/generate/stream")
@limiter.limit("10/minute")
async def generate_stream(request: Request, body: PromptRequest):
    # Проверяем безопасность промпта
    check_for_prompt_injection(body.prompt)

    async def token_generator():
        # Имитируем генерацию ответа нейросетью
        response_text = f"Привет! Это потоковый ответ на ваш запрос: «{body.prompt}». Я генерирую этот текст токен за токеном, прямо как настоящая большая языковая модель!"
        
        # Разбиваем текст на слова и отдаем их с небольшой задержкой
        words = response_text.split()
        for word in words:
            yield f"{word} "
            await asyncio.sleep(0.1)

    return StreamingResponse(token_generator(), media_type="text/event-stream")