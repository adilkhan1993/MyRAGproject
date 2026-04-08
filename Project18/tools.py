import warnings
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import logging

# Отключаем логирование предупреждений, чтобы не видеть желтый текст
logging.getLogger("duckduckgo_search").setLevel(logging.ERROR)

# --- 1. ИНСТРУМЕНТ ПОИСКА ---
@tool
def search_internet(query: str) -> str:
    """
    Ищет информацию в интернете по заданному запросу. 
    Возвращает список результатов с заголовками, кратким описанием и ссылками (URL).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
        if not results:
            # ЖЕСТКАЯ ИНСТРУКЦИЯ АГЕНТУ ПРИ БЛОКИРОВКЕ:
            return "ВНИМАНИЕ: Поисковик временно заблокировал запросы (пустой ответ). БОЛЬШЕ НЕ ИСПОЛЬЗУЙ ИНСТРУМЕНТ ПОИСКА. Сразу переходи к написанию отчета на основе того, что уже успел найти, или своих общих знаний."
            
        formatted_results = []
        for i, res in enumerate(results, 1):
            formatted_results.append(
                f"{i}. Заголовок: {res.get('title', 'Без заголовка')}\n"
                f"   Ссылка: {res.get('href', '')}\n"
                f"   Описание: {res.get('body', '')}\n"
            )
            
        return "\n".join(formatted_results)
    except Exception as e:
        # ЗАЩИТА ОТ ЦИКЛОВ ПРИ ОШИБКЕ:
        return f"ОШИБКА ПОИСКА: {str(e)}. ПРЕКРАТИ ПОИСКИ и пиши итоговый отчет."

# --- 2. ИНСТРУМЕНТ ПАРСИНГА СТРАНИЦ ---
@tool
def scrape_website(url: str) -> str:
    """
    Загружает содержимое веб-страницы по указанному URL и извлекает из нее текст.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        clean_text = text[:10000] 
        
        if not clean_text:
            return "Не удалось извлечь текст (пусто или нужен JavaScript). Возьми другую ссылку."
            
        return clean_text
        
    except requests.exceptions.Timeout:
        return f"Ошибка: сайт {url} грузится слишком долго (тайм-аут). Попробуй другую ссылку."
    except Exception as e:
        return f"Ошибка при чтении страницы {url}: {str(e)}. Попробуй другую ссылку."