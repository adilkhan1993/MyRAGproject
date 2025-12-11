from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .vectorstore import load_vectorstore

def format_docs(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        parts.append(f"Страница {page}:\n{d.page_content}")
    return "\n\n".join(parts)

def create_rag_chain():
    # Используем вашу Gemma 3
    llm = OllamaLLM(model="gemma3:1b")

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "Ты — помощник. Отвечай ТОЛЬКО на основе контекста PDF.\n"
        "Если информации нет, скажи: 'В документе этого нет'.\n"
        "Контекст:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever | format_docs}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def ask_question(chain, retriever, query: str):
    answer = chain.invoke(query)
    sources = retriever.invoke(query)
    return answer, sources