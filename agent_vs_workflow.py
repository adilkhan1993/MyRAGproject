import os
import re
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- ИМПОРТ ПОИСКОВИКА НАПРЯМУЮ ---
try:
    from duckduckgo_search import DDGS
except ImportError:
    print("❌ Ошибка: библиотека duckduckgo-search не найдена. Введите: pip install -U duckduckgo-search")
    sys.exit()

# 1. ЗАГРУЗКА
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# --- ФУНКЦИЯ ПОИСКА (Инструмент) ---
def run_search_tool(query):
    """Ищет информацию в DuckDuckGo напрямую."""
    try:
        # Ищем 1 результат
        results = DDGS().text(query, max_results=1)
        if results:
            return results[0]['body']
        return "Ничего не найдено."
    except Exception as e:
        return f"Ошибка поиска: {e}"

# --- ЧАСТЬ 1: ЖЕСТКИЙ WORKFLOW (Конвейер) ---
def run_workflow(city):
    print(f"\n⚙️  ЗАПУСК WORKFLOW для города: {city}...")
    
    # Шаг 1: Погода (Просто спрашиваем GPT, так как это жесткий скрипт)
    print("   1. Определяю погоду (GPT)...")
    weather_tmpl = PromptTemplate.from_template("Какая обычно погода в городе {city} в это время года? Кратко.")
    weather_chain = weather_tmpl | llm | StrOutputParser()
    weather = weather_chain.invoke({"city": city})
    print(f"      Результат: {weather[:50]}...")

    # Шаг 2: Достопримечательности
    print("   2. Ищу места...")
    sights_tmpl = PromptTemplate.from_template("Напиши топ-3 достопримечательности в городе {city}. Просто список.")
    sights_chain = sights_tmpl | llm | StrOutputParser()
    sights = sights_chain.invoke({"city": city})
    print(f"      Результат: {sights[:50]}...")

    # Шаг 3: Итог
    print("   3. Составляю план...")
    final_tmpl = PromptTemplate.from_template(
        "Ты гид. Город: {city}.\nПогода: {weather}\nМеста: {sights}\nСоставь план на день."
    )
    final_chain = final_tmpl | llm | StrOutputParser()
    
    final_plan = final_chain.invoke({
        "city": city,
        "weather": weather,
        "sights": sights
    })
    
    print("\n📝 ИТОГ WORKFLOW:\n" + final_plan)

# --- ЧАСТЬ 2: АВТОНОМНЫЙ АГЕНТ (Цикл) ---
def run_agent(city):
    print(f"\n🕵️‍♂️  ЗАПУСК АГЕНТА (Real Search) для города: {city}...")
    
    # Системная инструкция
    system_prompt = f"""
    Ты умный турагент. Твоя задача: спланировать поездку в город {city}.
    
    У тебя есть инструмент: [SEARCH] - поиск в интернете.
    
    Правила:
    1. Сначала узнай ТЕКУЩУЮ погоду через [SEARCH] (запрос 'weather in {city}').
    2. Если погода плохая, ищи музеи. Если хорошая — парки.
    3. Чтобы воспользоваться инструментом, напиши строго: Action: [SEARCH] "твой запрос"
    4. Когда будешь готов дать ответ, напиши: Final Answer: твой ответ.
    
    Действуй пошагово. Не придумывай погоду, ищи её!
    """
    
    conversation_history = system_prompt
    max_steps = 6
    step = 0
    
    # ВОТ ЗДЕСЬ БЫЛА ОШИБКА ОТСТУПОВ. ТЕПЕРЬ ВСЁ РОВНО:
    while step < max_steps:
        step += 1
        
        # 1. Мысль (Think)
        # Отправляем историю сообщений как один текст
        response = llm.invoke(conversation_history).content
        print(f"\n🤖 (Мысль): {response}")
        
        # Добавляем ответ модели в историю
        conversation_history += f"\n{response}"
        
        # 2. Проверка на финал
        if "Final Answer:" in response:
            break
            
        # 3. Действие (Act)
        # Ищем команду в ответе модели
        match = re.search(r'Action: \[SEARCH\] "(.*?)"', response)
        
        if match:
            search_query = match.group(1)
            print(f"🔎 (Действие): Ищу в интернете: '{search_query}'...")
            
            # Выполняем поиск
            observation = run_search_tool(search_query)
            print(f"👀 (Наблюдение): {observation[:100]}...") 
            
            # Добавляем результат поиска в историю
            conversation_history += f"\nObservation: {observation}\n"
        else:
            # Если агент ничего не ищет, просто продолжаем цикл
            continue

# --- ГЛАВНЫЙ БЛОК ---
if __name__ == "__main__":
    target_city = "London" 
    
    print("="*50)
    print("СРАВНЕНИЕ АРХИТЕКТУР (Project 12)")
    print("="*50)
    
    # 1. Запуск жесткого сценария
    try:
        run_workflow(target_city)
    except Exception as e:
        print(f"Ошибка Workflow: {e}")

    print("\n" + "="*50 + "\n")
    
    # 2. Запуск умного агента
    try:
        run_agent(target_city)
    except Exception as e:
        print(f"Ошибка Agent: {e}")