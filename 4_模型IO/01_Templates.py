'''from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)'''

#---------------------PromptTemplate---------------------
# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate

# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
# 创建方式1：from_template（）构建
prompt = PromptTemplate.from_template(template) 
# 打印LangChain提示模板的内容
price = 100 
flower_name = "玫瑰"
print(prompt.format(price=price, flower_name=flower_name))


# 创建方式2：PromTemplate（）构建
prompt2 = PromptTemplate(
    input_variables= ["price", "flower_name"],
    template= template)
price = 200
flower_name = "郁金香"
print(prompt2.format(price=price, flower_name=flower_name))


#---------------------Model---------------------    
# 设置OpenAI API Key
import os
#os.environ["OPENAI_API_KEY"] = '你的Open AI API Key'
os.environ["LLM_MODELEND"] = 'Doubao-pro-32k'
os.environ["OPENAI_BASE_URL"] = 'https://a0ai-api.zijieapi.com/api/llm/v1'

#----------------DOUBAO----------------
# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
# 创建模型实例
model = ChatOpenAI(model=os.environ["LLM_MODELEND"])
# 输入提示
input = prompt.format(flower_name=["玫瑰","郁金香"], price='50')
# 得到模型的输出
output = model.predict(input)
# output = model.invoke(input)
# 打印输出内容
print("------Doubao-pro-32k------")
print(output)  


#------------------Qwen----------------
# 导入LangChain中的huggingface_hub模型接口
# from huggingface_hub import InferenceClient # huggingface_hub原始接口
from langchain_huggingface import HuggingFaceEndpoint

# 创建模型实例
modelQwen = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-Coder-32B-Instruct")
# 输入提示
input = prompt.format(flower_name=["rose"], price='50')
# 得到模型的输出
output = model.predict(input)
# output = model.invoke(input)
# 打印输出内容

print("------hugging face model------")
print(output)