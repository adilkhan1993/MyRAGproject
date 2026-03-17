from crewai import Crew, Process
from agents import researcher, writer, editor
from tasks import research_task, write_task, edit_task

# Создаем команду
content_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential, 
    verbose=True
)