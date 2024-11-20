## LangChain系统安装与快速入门

---
### 1、LangChain
LangChain 是一个全方位的、基于大语言模型这种预测能力的应用开发工具，它的灵活性和模块化特性使得处理语言模型变得极其简便。

---
### 2、LangChain安装

```bash
# 基本安装
pip install langchain

# 需要包括常用开源LLM库时
pip install langchain[llms]

# 更新到最新版本
pip install --upgrade langchain

# 从源代码安装
pip install -e
```
LangChain的接口文档：[API 文档]()

---
### 3、OpenAI API
LangChain支持的两类模型：
- Chat Model， gpt-3.5-turbo和GPT-4等。
- Text Model， GPT-3,有专门训练出来做文本嵌入的text-embedding-ada-002,也有专门做相似度比较的模型，如text-similarity-curie-001。

---
### 4、调用Text模型
```python
# 直接调用Text模型 注意：已弃用
import os
from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
# os.environ["OPENAI_BASE_URL"] = 'OpenAI 的 API URL'

client = OpenAI()

response = client.completions.create(
    model=os.environ.get("LLM_MODELEND"),
    temperature=0.5,
    max_tokens=100,
    prompt="请给我的花店起个名",
)

print(response.choices[0].text.strip())
```
```python
# 使用LangChain调用Text模型
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
```

#### 使用步骤：
第一，注册API KEY
第二，使用`pip install openai`安装OpenAI库
第三，导入OpenAI API Key
```bash
# 建议：key保存在操作系统的环境变量中，Linux：
export OPEN_API_KEY='your key'
```
第四，导入OpenAI库，创建一个Client
```python
from openai import OpenAI
client = OpenAI()
```
第五，指定gpt-3.5-turbo-instruct，调用completions方法，返回结果。
```python

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  temperature=0.5,
  max_tokens=100,
  prompt="请给我的花店起个名")

```
控制输入内容样式的常见参数：
![alt text](..\0_images\2\image.png ){ width=600 height= }
第六，打印输出模型回答内容
```
print(response.choices[0].text.strip())
```
Test模型，响应对象主要字段包括：
![alt text](..\0_images\2\image2.png){ width=600 height= }

---
### 5、调用Chat模型
```python
# 直接调用Chat模型
import os
from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
# os.environ["OPENAI_BASE_URL"] = 'OpenAI 的 API URL'

client = OpenAI()

response = client.chat.completions.create(
    model=os.environ.get("LLM_MODELEND"),
    messages=[
        {"role": "system", "content": "You are a creative AI."},
        {"role": "user", "content": "请给我的花店起个名"},
    ],
    temperature=0.8,
    max_tokens=600,
)

print(response.choices[0].message.content)
```
```python
# 使用Langchain调用Chat模型
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'
# os.environ["OPENAI_BASE_URL"] = 'OpenAI 的 API URL'
# os.environ["LLM_MODELEND"] = 'Doubao-pro-32k'

chat = ChatOpenAI(model=os.environ.get("LLM_MODELEND"), temperature=0.8, max_tokens=600)

messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名"),
]
response = chat(messages)
print(response)
```
**可以看出LangChain对数据格式和OpenAI模型做了封装，使输入与输出更加简洁**

这里两个专属于Chat模型的概念，一个是**消息**，一个是**角色**。
- **消息**：传入模型的提示，每个消息都有一个role（可以是system、user或assistant） 和 content。
- **角色**：systtem（主要用于设定上下文，帮助模型理解对话中的角色和任务）、user（从用户出发，包含用户想要模型回答的请求）、assistant（模型的回复）
```json
// 一个典型的response对象
{
 'id': 'chatcmpl-2nZI6v1cW9E3Jg4w2Xtoql0M3XHfH',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-4',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': '你的花店可以叫做"花香四溢"。'
     },
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}
```
下面是个字段的含义：
![alt text](..\0_images\2\image3.png){ width=600 height= }
**相比于Text模型的响应结构，只是choices字段中Text换成了Message**，`response['choices'][0]['message']['content']`取模型

### Chat模型vsText模型
相较于Text模型，Chat模型的设计更适合处理对话或者多轮次交互的情况。这是因为它可以接受一个消息列表作为输入，而不仅仅是一个字符串。这个消息列表可以包含system、user和assistant的历史信息，从而在处理交互式对话时提供更多的上下文信息。

这种设计的主要优点包括：

- 对话历史的管理：通过使用Chat模型，你可以更方便地管理对话的历史，并在需要时向模型提供这些历史信息。例如，你可以将过去的用户输入和模型的回复都包含在消息列表中，这样模型在生成新的回复时就可以考虑到这些历史信息。
- 角色模拟：通过system角色，你可以设定对话的背景，给模型提供额外的指导信息，从而更好地控制输出的结果。当然在Text模型中，你在提示中也可以为AI设定角色，作为输入的一部分。