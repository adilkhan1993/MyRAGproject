import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Убираем спам-предупреждения от LangGraph
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Загружаем .env
load_dotenv()

# ==========================================
# 🛡️ ЖЕЛЕЗОБЕТОННОЕ ВКЛЮЧЕНИЕ LANGSMITH
# ==========================================
# ПРИНУДИТЕЛЬНО прописываем настройки прямо в коде, чтобы перебить любые сбои терминала
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Project19_Debugging"

# ==========================================
# 🛑 НАМЕРЕННО СЛОМАННЫЕ ИНСТРУМЕНТЫ 
# ==========================================

@tool
def get_employee_salary(employee_id: int) -> str:
    """Возвращает зарплату сотрудника по его ID (строго число!)."""
    # Баг 1: Если LLM передаст имя (строку) вместо ID (числа), LangChain выбросит ошибку валидации
    db = {101: "150,000 руб.", 102: "200,000 руб."}
    return db.get(employee_id, "Сотрудник не найден")

@tool
def search_internal_wiki(query: str) -> str:
    """Ищет информацию во внутренней корпоративной вики-системе."""
    # Баг 2: Имитация сломанного API. Инструмент заставит агента зациклиться.
    return "Ошибка 503: Сервер перегружен. Пожалуйста, повторите запрос."

# ==========================================
# 🤖 ИНИЦИАЛИЗАЦИЯ АГЕНТА
# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_employee_salary, search_internal_wiki]

# Баг 3: Токсичный промпт, провоцирующий ошибки
BAD_PROMPT = """Ты корпоративный ИИ-ассистент.
ПРАВИЛО 1: Если не знаешь ответ или инструмент выдал ошибку, просто верни пустую строку. Ничего не объясняй пользователю!
ПРАВИЛО 2: Пользователи часто спрашивают про зарплату по имени. Обязательно сначала попробуй передать в инструмент ИМЯ, а не ID."""

agent_executor = create_react_agent(llm, tools)

def main():
    print("🚀 Запускаем сломанного агента (версия v1-broken)...\n")
    
    # 🎯 ВАЖНО ДЛЯ LangSmith: Добавляем тег 'v1-broken', как требует задание
    config = {
        "tags": ["v1-broken"],
        "recursion_limit": 5 # Спасаем ваш баланс токенов от бесконечного цикла
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
                    ("system", BAD_PROMPT),
                    ("user", q)
                ]
            }, config=config)
            
            print(f"🤖 Ответ ИИ: '{response['messages'][-1].content}'\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА АГЕНТА: {e}\n")
            print("-" * 50)

if __name__ == "__main__":
    main()