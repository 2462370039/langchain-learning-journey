## Model I/O 
对模型的使用过程可分为三部分：**输入提示**、**调用模型**和**输出解析**。
![alt text](..\0_images\4\image.png){width=600}

---
### 提示模板
相关：吴恩达老师的[提示工程课程](https://learn.deeplearning.ai/login?redirect_course=chatgpt-prompt-eng)

PromptTemplate的from_template方法就是将一个原始的模板字符串转化为一个更丰富、更方便操作的PromptTemplate对象，这个对象就是LangChain中的提示模板。
```python
# 导入LangChain中的提示模板
from langchain.prompts.prompt import PromptTemplate
# 创建原始模板
template = """您是一位专业的鲜花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
"""
# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template) 
# 打印LangChain提示模板的内容
print(prompt)
```

---
### 语言模型
使用LangChain调用各种语言模型
#### 1、OpenAI的模型
```python
# 设置OpenAI API Key
import os
os.environ["OPENAI_API_KEY"] = '你的Open AI API Key'

# 导入LangChain中的OpenAI模型接口
from langchain_openai import OpenAI
# 创建模型实例
model = OpenAI(model_name='gpt-3.5-turbo-instruct')
# 输入提示
input = prompt.format(flower_name=["玫瑰"], price='50')
# 得到模型的输出
output = model.invoke(input)
# 打印输出内容
print(output)  
```
#### 2、Doubao模型
```python
# 设置OpenAI API Key
import os
#os.environ["OPENAI_API_KEY"] = '你的Open AI API Key'
os.environ["LLM_MODELEND"] = 'Doubao-pro-32k'
os.environ["OPENAI_BASE_URL"] = 'https://a0ai-api.zijieapi.com/api/llm/v1'

# 导入LangChain中的OpenAI模型接口
from langchain_openai import ChatOpenAI
# 创建模型实例
model = ChatOpenAI(model=os.environ["LLM_MODELEND"])
# 输入提示
input = prompt.format(flower_name=["玫瑰","郁金香"], price='50')
# 得到模型的输出
output = model.invoke(input)
# 打印输出内容
print("------Doubao-pro-32k------")
print(output)  
```
打印内容：
```
------Doubao-pro-32k------
content='仅需 50 元，即可拥有娇艳欲滴的玫瑰与优雅迷人的郁金香，双色鲜花，双倍浪漫，让美好瞬间绽放！' 

additional_kwargs={'refusal': None} 

response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 48, 'total_tokens': 79, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'Doubao-pro-32k', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}

id='run-f9db8f22-153c-4209-a3fc-8959a658b63d-0' 

usage_metadata={'input_tokens': 48, 'output_tokens': 31, 'total_tokens': 79, 'input_token_details': {}, 'output_token_details': {}}
```

#### 3、huggingface_hub中的开源模型
```python
# 导入LangChain中的huggingface_hub模型接口
# from huggingface_hub import InferenceClient # huggingface_hub原始接口
from langchain_huggingface import HuggingFaceEndpoint

# 创建模型实例
modelQwen = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-Coder-32B-Instruct")
# 输入提示
input = prompt.format(flower_name=["rose"], price='50')
# 得到模型的输出
output = model.invoke(input)
# 打印输出内容

print("------hugging face model------")
print(output)
```
打印内容：
```
------hugging face model------
content='50 元的 rose：娇艳欲滴的玫瑰，象征着热烈的爱情与真挚的情感，50 元即可拥有这份浪漫与美好，让它为你的生活增添一抹绚丽色彩。' 

additional_kwargs={'refusal': None} 

response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 45, 'total_tokens': 88, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'Doubao-pro-32k', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None} id='run-7509c64b-b04c-4122-9db8-90ddf6dd0e2d-0' 

usage_metadata={'input_tokens': 45, 'output_tokens': 43, 'total_tokens': 88, 'input_token_details': {}, 'output_token_details': {}}
```

LangChain的**优势**:我们只需要定义一次模板，就可以用它来生成各种不同的提示。
因此，使用LangChain和提示模板的好处是：
- 代码的可读性：使用模板的话，提示文本更易于阅读和理解，特别是对于复杂的提示或多变量的情况。
- 可复用性：模板可以在多个地方被复用，让你的代码更简洁，不需要在每个需要生成提示的地方重新构造提示字符串。
- 维护：如果你在后续需要修改提示，使用模板的话，只需要修改模板就可以了，而不需要在代码中查找所有使用到该提示的地方进行修改。
- 变量处理：如果你的提示中涉及到多个变量，模板可以自动处理变量的插入，不需要手动拼接字符串。
- 参数化：模板可以根据不同的参数生成不同的提示，这对于个性化生成文本非常有用。

### 输出解析
在开发具体应用的过程中，很明显**我们不仅仅需要文字，更多情况下我们需要的是程序能够直接处理的、结构化的数据。** ————使用LangChain中的解析器处理响应
```python
#---------------------StructuredOutputParser---------------------
# 导入结构化输出解析器和ResponseSchema
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# 定义我们想要接收的响应模式
response_schemas = [
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="问什么要这样写这个文案")
]
# 创建输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 获取格式化指示
format_instructions = output_parser.get_format_instructions()
# 根据原始模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, 
                partial_variables={"format_instructions": format_instructions}) 

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 创建一个空的DataFrame用于存储结果
import pandas as pd
df = pd.DataFrame(columns=["flower", "price", "description", "reason"]) # 先声明列名

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower_name=flower, price=price)

    # 获取模型的输出
    # output = model.invoke(input)
    output = model.predict(input) 

    # 解析模型的输出（这是一个字典结构）
    parsed_output = output_parser.parse(output)

    # 在解析后的输出中添加“flower”和“price”
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output  

# 打印字典
print(df.to_dict(orient='records'))

# 保存DataFrame到CSV文件
df.to_csv(".\\4_模型IO\\flowers_with_descriptions.csv", index=False)
```

返回结果：
```
[
{'flower': '玫瑰', 'price': '50', 'description': '50 元的玫瑰，绽放着浪漫与爱情的芬芳，为你的生活增添一抹绚丽的色彩。', 'reason': '这个文案突出了玫瑰的浪漫和爱情象征意义，同时强调了其价格实惠，能够吸引消费者的注意。'}, 
{'flower': '百合', 'price': '30', 'description': '纯洁百合，仅需 30 元，让清新与美好绽放于你的生活。', 'reason': '突出百合的纯洁特质，强调价格实惠，吸引顾客购买，同时表达出百合能为生活带来美好和清新的感觉。'}, 
{'flower': '康乃馨', 'price': '20', 'description': '仅需 20 元，即可拥有一束温馨的康乃馨，为生活增添一抹温暖的色彩。', 'reason': '突出价格实惠，同时强调康乃馨能带来的温馨氛围，吸引消费者购买。'}
]
```

总之，**输出解析器（Output Parser）的作用**是将模型生成的原始输出转换为结构化的数据格式，以便于进一步的处理和分析。