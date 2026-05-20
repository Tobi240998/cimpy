from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:32b",
    base_url="http://localhost:11434",
    temperature=0.0, 
    reasoning=False,
)

response = llm.invoke("Was ist 2+2?")
print(response.content)
print(response.additional_kwargs)
print(response.usage_metadata)