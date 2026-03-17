import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import get_weather, get_news

# Загружаем ключи из .env
load_dotenv()

# Инициализируем LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Собираем наши инструменты в один список
tools = [get_weather, get_news]

# Создаем агента вообще без модификаторов! Оставляем только самое необходимое.
agent_executor = create_react_agent(llm, tools)