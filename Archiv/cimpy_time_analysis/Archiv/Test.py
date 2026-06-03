from langchain_llm import get_llm
from langchain_core.messages import HumanMessage

llm = get_llm()
response = llm.invoke([HumanMessage(content="Sag kurz Hallo")])
print(response.content)
