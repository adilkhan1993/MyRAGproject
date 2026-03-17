import os
from dotenv import load_dotenv

# 1. СНАЧАЛА ЗАГРУЖАЕМ КЛЮЧИ!
load_dotenv()

# 2. И ТОЛЬКО ПОТОМ ИМПОРТИРУЕМ АГЕНТОВ
from crew import content_crew

def main():
    print("==================================================")
    print("🚀 ЗАПУСК: ПРОЕКТ 16 (Professional Edition)")
    print("Мультиагентная система CrewAI для создания контента")
    print("==================================================\n")

    topic = input("Введите тему для статьи: ")

    print(f"\n[INFO] Агенты приступают к работе над темой: '{topic}'...\n")

    # Запускаем конвейер
    inputs = {'topic': topic}
    result = content_crew.kickoff(inputs=inputs)

    print("\n==================================================")
    print("✅ ПРОЕКТ 16 УСПЕШНО ЗАВЕРШЕН!")
    print("Результат сохранен в папке 'output/final_article.md'")
    print("==================================================")

if __name__ == "__main__":
    main()