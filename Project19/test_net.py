import os
import requests
from dotenv import load_dotenv

load_dotenv()

print("🌐 ТЕСТИРОВАНИЕ СЕТИ 🌐\n")

# 1. Проверяем OpenAI
print("1. Стучимся в OpenAI...")
try:
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    r = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=5)
    print(f"✅ OpenAI ответил! Статус: {r.status_code}")
except Exception as e:
    print(f"❌ ОШИБКА OpenAI: {type(e).__name__} - {str(e)[:100]}")

print("-" * 40)

# 2. Проверяем LangSmith
print("2. Стучимся в LangSmith...")
try:
    r = requests.get("https://api.smith.langchain.com", timeout=5)
    print(f"✅ LangSmith ответил! Статус: {r.status_code}")
except Exception as e:
    print(f"❌ ОШИБКА LangSmith: {type(e).__name__} - {str(e)[:100]}")