import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
os.environ["OPENAI_BASE_URL"] = 'https://a0ai-api.zijieapi.com/api/llm/v1'
os.environ["LLM_MODELEND"] = 'Doubao-pro-32k'

chat = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0.8, max_tokens=600)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名"),
]

response = chat(messages)
print(response)