import os

#os.environ["OPENAI_API_KEY"] = ''
os.environ["OPENAI_BASE_URL"] = 'https://a0ai-api.zijieapi.com/api/llm/v1'
os.environ["LLM_MODELEND"] = 'Doubao-pro-32k'

from langchain_openai import ChatOpenAI, OpenAI

# llm = OpenAI(model_name="gpt-3.5-turbo-instruct",max_tokens=200)
llm = ChatOpenAI(model=os.environ.get("LLM_MODELEND"))

text = llm.predict("请给我写一句PeRo绝地求生电子竞技俱乐部的中文宣传语")
print(text)
