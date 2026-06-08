import os
import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPEN_ROUTER")

async def main():

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mcp_server.py"],
                "transport": "stdio"
            }
        }
    )

    tools = await client.get_tools()

    llm = init_chat_model(
        "openrouter:openai/gpt-oss-120b:free"
    )

    agent = create_react_agent(
        llm,
        tools
    )

    response = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "What is 25 multiplied by 4?"
                )
            ]
        }
    )

    print(
        response["messages"][-1].content
    )


asyncio.run(main())