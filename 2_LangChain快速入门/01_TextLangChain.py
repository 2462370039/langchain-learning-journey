import os
from langchain_openai import ChatOpenAI


os.environ["OPENAI_BASE_URL"] = 'https://a0ai-api.zijieapi.com/api/llm/v1'
os.environ["LLM_MODELEND"] = 'Doubao-pro-32k'

llm = ChatOpenAI(
    model=os.environ.get("LLM_MODELEND"),
    temperature=0.8,
    max_tokens=600,
)
response = llm.predict("请给我的花店起个名")

print(response)