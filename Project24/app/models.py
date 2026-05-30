from pydantic import BaseModel, Field, field_validator

class PromptRequest(BaseModel):
    # Ограничиваем длину промпта до 2000 символов
    prompt: str = Field(..., max_length=2000, description="Текст запроса к нейросети")
    # Параметры генерации тоже берем под контроль
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Кастомная проверка: промпт не должен быть пустым или состоять только из пробелов
    @field_validator('prompt')
    @classmethod
    def validate_prompt_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Промпт не может быть пустым или состоять только из пробелов")
        return value