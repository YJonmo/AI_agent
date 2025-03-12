import os
import io
import torch
from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from langchain_core.tools import Tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from wikipedia import summary
from PIL import Image
from langchain.chains import LLMMathChain

from .utils import (
    cifar_transforms,
    cifar_model,
    cifar_classes,
    device
)

app = FastAPI()

os.environ["NVIDIA_API_KEY"] = "your-api-key-here"  # Replace it with your actual API key obtained from:
# https://build.nvidia.com/mistralai/mixtral-8x7b-instruct?snippet_tab=Python&signin=true&api_key=true

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a Chat model
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

llm_math = LLMMathChain.from_llm(llm=llm)
def calculator_tool_wrapper(input_text: str) -> str:
    try:
        result = llm_math.run(input_text)
        return str(result)  # Convert the output to a string
    except Exception as e:
        return f"Error: {str(e)}"
    
class SearchWikipedia(BaseTool):
    name:str = "Wikipedia"
    description:str = """
        Searches Wikipedia and returns the summary of the first result.
        Use your knowledge first and if you were unable to answer to a general question that you don't know then use this tool.
        For calculations and math use the CalculatorTool and not Wikipedia!
    """
    def _run(self, query: str="don't do any thing", **kwargs)->str:
        try:
            # Limit to two sentences for brevity
            return summary(query, sentences=2)
        except Exception as e:        
            return f"I couldn't find any information on that. Here is the error message {e}"


@app.post("/image_classifier/")
async def image_classifier(file: UploadFile = File(...))->str:
    # Read and preprocess the image
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    input_tensor = cifar_transforms(image).to(device)
    with torch.no_grad():
        output = cifar_model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted = cifar_classes[predicted]

    chat_response = f"The ML model was able detect a {predicted} in the image."
    return chat_response

tools = [
    SearchWikipedia(),
    Tool(
        name="ImageClassifier",
        func=image_classifier,
        description=""""
            Explain in one sentence what you see in the image.
            Only use this tool when an image was uploaded.
        """,
    ),
    Tool(
        name="Calculator",
        func=calculator_tool_wrapper,
        description="""
            Useful for when you need to answer questions about math.
            This tool is only for math questions and nothing else.
            Formulate the input as python code.
        """,
    ),
]

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)


chat_history = []  # Collect chat history here (a sequence of messages)
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Calculator, image classifier, and Wikipedia."
chat_history.append(HumanMessage(content=""))
chat_history.append(AIMessage(content=initial_message))


templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Chat Loop to interact with the user
@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    message = data["message"]
    response = agent_executor.invoke({"input": message, "chat_history": chat_history})

    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=str(response['output'])))
    return {"response": response['output']}


