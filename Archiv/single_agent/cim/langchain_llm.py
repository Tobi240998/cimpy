from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="qwen3:32b",
        base_url="http://localhost:11434",
        temperature=0.0,
        streaming=False,
        timeout=180, 
        reasoning=False
    )