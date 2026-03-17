import os
import requests
from langchain_core.tools import tool

# --- 1. ИНСТРУМЕНТ ПОГОДЫ (OpenWeatherMap) ---
@tool
def get_weather(city: str) -> str:
    """
    Возвращает текущую погоду для заданного города.
    Используйте этот инструмент, когда пользователь спрашивает о погоде или климате.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Ошибка: Ключ WEATHER_API_KEY не найден."
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric", # Чтобы температура была в градусах Цельсия
        "lang": "ru"       # Ответ на русском языке
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Проверка на ошибки (например, если город не найден)
        data = response.json()
        
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        
        return f"Текущая погода в городе {city}: {temp}°C, {description}. Влажность: {humidity}%."
    except requests.exceptions.RequestException as e:
        return f"К сожалению, не удалось получить погоду для города {city}. Ошибка: {e}"

# --- 2. ИНСТРУМЕНТ НОВОСТЕЙ (NewsAPI) ---
@tool
def get_news(query: str) -> str:
    """
    Возвращает топ-3 последних новостей по заданному запросу или теме.
    Используйте этот инструмент, когда пользователь просит найти новости, события или информацию о происходящем.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "Ошибка: Ключ NEWS_API_KEY не найден."
        
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "ru",
        "sortBy": "publishedAt",
        "pageSize": 3 # Берем только 3 свежие новости, чтобы не перегружать агента
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get("articles", [])
        if not articles:
            return f"По запросу '{query}' новостей не найдено."
            
        result = f"Свежие новости по теме '{query}':\n"
        for i, article in enumerate(articles, 1):
            title = article.get("title", "Без заголовка")
            source = article.get("source", {}).get("name", "Неизвестный источник")
            result += f"{i}. {title} (Источник: {source})\n"
            
        return result
    except requests.exceptions.RequestException as e:
        return f"К сожалению, не удалось получить новости по запросу '{query}'. Ошибка: {e}"