from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="phi3:mini",
        base_url="http://localhost:11434",
        temperature=0.0,
        streaming=False,
        timeout=180
    )