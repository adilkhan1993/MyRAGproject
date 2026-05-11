import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# ЖЕЛЕЗОБЕТОННОЕ ВКЛЮЧЕНИЕ LANGSMITH
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Project19_Debugging"

# ==========================================
# 🛠 ИСПРАВЛЕННЫЕ ИНСТРУМЕНТЫ 
# ==========================================

@tool
def get_employee_salary(employee_id: int) -> str:
    """Возвращает зарплату сотрудника. ВАЖНО: Принимает ТОЛЬКО числовой ID (например, 101). Не передавай имена!"""
    db = {101: "150,000 руб.", 102: "200,000 руб."}
    return db.get(employee_id, "Сотрудник с таким ID не найден.")

@tool
def search_internal_wiki(query: str) -> str:
    """Ищет информацию во внутренней корпоративной вики-системе."""
    return "Ошибка 503: Сервер временно недоступен. Ведутся технические работы."

# ==========================================
# 🤖 ИНИЦИАЛИЗАЦИЯ ИСПРАВЛЕННОГО АГЕНТА
# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_employee_salary, search_internal_wiki]

GOOD_PROMPT = """Ты полезный корпоративный ИИ-ассистент.
ПРАВИЛО 1: База зарплат работает ТОЛЬКО по числовым ID. Если пользователь спрашивает по имени (например, "Иван"), вежливо попроси его назвать числовой ID.
ПРАВИЛО 2: Если какой-либо инструмент возвращает ошибку (например, сервер недоступен), честно и вежливо сообщи об этом пользователю."""

agent_executor = create_react_agent(llm, tools)

def main():
    print("🚀 Запускаем исцеленного агента (версия v2-fixed)...\n")
    
    # 🎯 МЕНЯЕМ ТЕГ НА v2-fixed
    config = {
        "tags": ["v2-fixed"],
        "recursion_limit": 5 
    }

    queries = [
        "Какая зарплата у Ивана?",
        "Найди в вики информацию про новый график отпусков."
    ]

    for q in queries:
        print(f"👤 Пользователь: {q}")
        try:
            response = agent_executor.invoke({
                "messages": [
                    ("system", GOOD_PROMPT),
                    ("user", q)
                ]
            }, config=config)
            
            print(f"🤖 Ответ ИИ: {response['messages'][-1].content}\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")
            print("-" * 50)

if __name__ == "__main__":
    main()