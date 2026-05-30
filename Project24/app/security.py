from fastapi import HTTPException

# Список подозрительных фраз, которые мы будем блокировать
FORBIDDEN_PATTERNS = [
    "ignore previous instructions",
    "system:",
    "забудь все инструкции",
    "переключи роль"
]

def check_for_prompt_injection(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in prompt_lower:
            # Если нашли запрещенку - выдаем ошибку 400 Bad Request
            raise HTTPException(
                status_code=400, 
                detail=f"Запрос отклонён: обнаружен подозрительный контент ({pattern})"
            )
    return prompt