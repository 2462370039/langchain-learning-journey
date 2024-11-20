
# ---- Part 0 导入所需要的类
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# ---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

cache_dir = "/tmp/cache"

os.environ["LLM_MODEL_4K_FUNCTION_CALL"] = "Doubao-pro-4k-function-call"

# 确保缓存目录存在
os.makedirs(cache_dir, exist_ok=True)

print("正在初始化图像字幕生成模型...")

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model, cache_dir=cache_dir)
# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model, cache_dir=cache_dir)
print("初始化图像字幕生成模型成功")


# ---- Part II 定义图像字幕生成工具类
class ImageCapTool(BaseTool):
    name = "Image captioner"
    description = "使用该工具可以生成图片的文字描述，需要传入图片的URL."

    def _run(self, url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


# ---- PartIII 初始化并运行LangChain智能体
# llm = OpenAI(temperature=0.2)
llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL_4K_FUNCTION_CALL"),
    temperature=0,
)
print("初始化大语言模型成功")
# 使用工具初始化智能体并运行
tools = [ImageCapTool()]
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
)

img_url = "https://imgxz.bizhi3.com/20240119/240119020253947.jpg"
# agent.run(input=f"{img_url}\n请创作合适的中文推广文案")
agent.invoke(input=f"图片链接如下：{img_url}\n 请为这张图创作合适的中文推广文案")


