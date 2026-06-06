# Previously wrong:
# - No tool validation
# - No execution loop
# - No structured tool responses

import json
from openai import OpenAI

client = OpenAI(
    api_key="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1"
)

messages = [
    {
        "role": "system",
        "content": """
You are an AI agent.

Rules:
- If a tool is needed, respond ONLY in JSON:
{
  "tool": "tool_name",
  "arguments": {}
}

- If final response:
{
  "final": "answer"
}
"""
    }
]


def get_weather(city: str):
    return f"Weather in {city}: 34C"


TOOLS = {
    "get_weather": get_weather
}


def execute_tool(tool_name, arguments):
    if tool_name not in TOOLS:
        raise Exception("Invalid tool")

    return TOOLS[tool_name](**arguments)


while True:
    user_input = input("You: ")

    messages.append({
        "role": "user",
        "content": user_input
    })

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages
        )

        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)

        except Exception:
            print("Model returned invalid JSON")
            break

        if "tool" in parsed:
            tool_name = parsed["tool"]
            arguments = parsed["arguments"]

            print(f"\n[TOOL CALL] {tool_name} {arguments}")

            try:
                tool_result = execute_tool(tool_name, arguments)

            except Exception as e:
                tool_result = f"Tool execution failed: {str(e)}"

            print(f"[TOOL RESULT] {tool_result}\n")

            messages.append({
                "role": "assistant",
                "content": content
            })

            messages.append({
                "role": "tool",
                "content": str(tool_result)
            })

            continue

        if "final" in parsed:
            answer = parsed["final"]

            print(f"AI: {answer}")

            messages.append({
                "role": "assistant",
                "content": content
            })

            break