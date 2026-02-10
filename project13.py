import os
import re
import datetime
import math
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. ЗАГРУЗКА
load_dotenv()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# --- ИНСТРУМЕНТЫ (TOOLS) ---

def get_current_date(query=None):
    """Возвращает текущую дату и день недели."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d (%A)")

def calculator(expression):
    """Вычисляет математическое выражение."""
    try:
        # Удаляем лишние слова, оставляем только цифры и знаки
        clean_expr = re.sub(r'[^0-9+\-*/()., mathsqrt]', '', expression)
        # Разрешаем использовать math.sqrt
        return str(eval(clean_expr, {"__builtins__": None}, {"math": math, "sqrt": math.sqrt}))
    except Exception as e:
        return f"Ошибка вычисления: {e}"

# Словарь инструментов для агента
tools_map = {
    "DATE": get_current_date,
    "CALCULATOR": calculator
}

# --- АВТОНОМНЫЙ АГЕНТ (ReAct Loop) ---
def run_autonomous_agent(user_query):
    print(f"\n🤖 ЗАПУСК АГЕНТА по задаче: \"{user_query}\"\n")
    
    # Системный промпт (Мозг агента) [cite: 61-62]
    system_prompt = f"""
    Ты умный помощник. Твоя задача — ответить на вопрос пользователя.
    
    У тебя есть инструменты:
    1. [DATE] - узнать текущую дату. (Аргумент не нужен)
    2. [CALCULATOR] - выполнить вычисление. (Пример: 341 * 5 или sqrt(100))
    
    ФОРМАТ ТВОИХ МЫСЛЕЙ:
    Question: Вопрос пользователя
    Thought: Твои рассуждения (что делать дальше?)
    Action: [ИМЯ_ИНСТРУМЕНТА] "значение"
    Observation: Результат работы инструмента
    ... (повторяй Thought/Action/Observation сколько нужно)
    Final Answer: Окончательный ответ пользователю.
    
    Вопрос: {user_query}
    """
    
    conversation_history = system_prompt
    max_steps = 10
    step = 0
    
    while step < max_steps:
        step += 1
        
        # 1. МЫСЛЬ (Thought)
        response = llm.invoke(conversation_history).content
        print(f"🧠 (Мысль): {response}")
        conversation_history += f"\n{response}"
        
        if "Final Answer:" in response:
            return # Завершаем работу
            
        # 2. ДЕЙСТВИЕ (Action)
        # Ищем паттерн: Action: [TOOL] "value"
        match = re.search(r'Action: \[(.*?)\] "(.*?)"', response)
        
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            
            print(f"🛠️ (Действие): Вызываю {tool_name} с параметром '{tool_input}'...")
            
            # Запускаем Python-функцию
            if tool_name in tools_map:
                try:
                    observation = tools_map[tool_name](tool_input)
                except Exception as e:
                    observation = f"Error: {e}"
            else:
                observation = "Ошибка: Такого инструмента нет."
                
            print(f"👀 (Наблюдение): {observation}\n")
            
            # Записываем результат в память агента
            conversation_history += f"\nObservation: {observation}\n"
        else:
            # Если агент забыл формат, напоминаем ему (скрытый механизм)
            if "Action:" in response:
                print("   (Агент ошибся в формате, пробую подтолкнуть...)")
                conversation_history += "\nSystem Note: Пожалуйста, используй формат Action: [TOOL_NAME] \"input\""
            continue

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Сложная задача из PDF 
    task = "Сколько дней осталось до Нового года (1 января 2027), и чему равен корень квадратный из этого числа?"
    
    run_autonomous_agent(task)