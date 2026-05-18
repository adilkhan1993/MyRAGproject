from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048, description="Текстовый промпт для генерации")

class GenerationResponse(BaseModel):
    generated_text: str