from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(model="phi3", temperature=0.0)

