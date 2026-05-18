from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.models import PromptRequest, GenerationResponse
from app.inference import inference_engine

# Загружаем модель при старте сервера
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        inference_engine.load_model()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
    yield
    # Здесь можно освобождать ресурсы при выключении (если нужно)
    print("Сервер остановлен.")

app = FastAPI(
    title="AI Text Generator API",
    description="API для генерации текста с использованием дообученной модели",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    model_loaded = inference_engine.model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "message": "Сервис работает" if model_loaded else "Сервис работает, но модель не загружена"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: PromptRequest):
    """Генерация текста по промпту"""
    try:
        # Логика передается в inference.py
        result = inference_engine.generate(request.prompt)
        return GenerationResponse(generated_text=result)
        
    except RuntimeError as e:
        # Ошибка 500: Внутренняя проблема (например, модель не успела загрузиться)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Ошибка 500: Любая другая непредвиденная ошибка генерации
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")