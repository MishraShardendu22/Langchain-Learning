# Gen-AI — Learning Repository

A comprehensive, hands-on learning repository exploring modern **Generative AI** concepts including **Retrieval-Augmented Generation (RAG)**, **LangChain**, **LangGraph**, **LLM Guardrails**, **Model Context Protocol (MCP)**, and **Tool Calling** with various LLM providers.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Concepts Covered](#core-concepts-covered)
   - [Retrieval-Augmented Generation (RAG)](#1-retrieval-augmented-generation-rag)
   - [LangChain](#2-langchain)
   - [LangGraph](#3-langgraph)
   - [Guardrails & Safety](#4-guardrails--safety)
   - [Model Context Protocol (MCP)](#5-model-context-protocol-mcp)
   - [Tool Calling](#6-tool-calling)
4. [Getting Started](#getting-started)
5. [Environment Setup](#environment-setup)
6. [Dependencies](#dependencies)

---

## Project Overview

This repository is an **experimental learning hub** for Generative AI engineering. It contains Jupyter notebooks and Python scripts that demonstrate:

- Building **RAG pipelines** from scratch (basic → advanced)
- Creating **AI agents** with LangChain and LangGraph
- Implementing **guardrails** for safe LLM deployment
- Using the **Model Context Protocol (MCP)** for tool integration
- **Tool calling** patterns with structured execution loops
- Multi-provider LLM orchestration (OpenRouter, Groq, Gemini)

---

## Repository Structure

```
Gen-AI/
├── main.py                         # Project entry point
├── pyproject.toml                  # Python project configuration
├── README.md                       # You are here
│
├── rag/
│   ├── rag_learn.ipynb            # Comprehensive RAG notebook
│   └── attention.pdf              # "Attention Is All You Need" paper (PDF)
│
├── lang-chain-graph/
│   ├── langchain.ipynb            # LangChain fundamentals
│   ├── langgraph.ipynb            # LangGraph fundamentals
│   └── README.md                  # LangChain-specific docs
│
├── mcp/
│   ├── main.py                    # MCP client (LangChain adapter)
│   └── mcp_server.py             # MCP server (FastMCP)
│
├── gaurdrails-gateway/
│   └── Gaurdrali_Gateway.ipynb   # Guardrails implementation
│
└── sample/
    └── tool-callin-sample.py     # Basic tool calling example
```

---

## Core Concepts Covered

### 1. Retrieval-Augmented Generation (RAG)

**Location:** `rag/rag_learn.ipynb`

RAG combines **information retrieval** with **LLM generation** to ground model responses in factual, external knowledge.

#### Topics Covered

| Concept | Description |
|---------|-------------|
| **Basic RAG Pipeline** | Retrieve → Augment → Generate loop |
| **Vector Stores (FAISS)** | Meta's FAISS for efficient similarity search |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` for semantic representations |
| **Document Loading** | `PyMuPDFLoader` for PDF ingestion |
| **Text Chunking** | `RecursiveCharacterTextSplitter` with configurable overlap |
| **Similarity Search** | Cosine similarity retrieval with `k` nearest neighbors |
| **MMR Search** | Maximum Marginal Relevance for diverse retrieval |
| **Hybrid Search** | BM25 (keyword) + Vector (semantic) combined retrieval |
| **Reranking** | Cohere Rerank API + CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| **Query Expansion** | LLM-generated sub-query expansion for better recall |
| **Parent-Child Chunking** | Small child chunks indexed → large parent chunks returned |
| **Streaming RAG** | Token-by-token streaming of generated answers |
| **Cited RAG** | Source-attributed answers with metadata tracking |
| **LangSmith Evaluation** | RAG pipeline correctness evaluation |
| **FAISS Persistence** | Save/load vector indexes to disk |
| **Chat History** | Contextual follow-up question handling |
| **Content Compression** | Query-relevant sentence extraction from chunks |

#### Architecture Diagram

```
User Query
    │
    ▼
[Retriever] ────→ Vector DB (FAISS)
    │                    │
    │              Embedding Model
    │                    │
    ▼                    ▼
Retrieved Documents ────→ Context Construction
    │
    ▼
[LLM] ────→ Grounded Answer
```

#### Key Code (Basic RAG Pipeline)

```python
def rag_pipeline(query: str):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""Use the context below to answer the question.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    response = llm.invoke(prompt)
    return {"query": query, "context": context, "answer": response.content}
```

---

### 2. LangChain

**Location:** `lang-chain-graph/langchain.ipynb`

LangChain is a framework for building LLM-powered applications with abstractions for models, tools, memory, agents, and chains.

#### Topics Covered

| Concept | Description |
|---------|-------------|
| **Chat Models** | Gemini (`ChatGoogleGenerativeAI`), OpenRouter (`ChatOpenRouter`) |
| **Model Initialization** | `init_chat_model()` for provider-agnostic setup |
| **Tool Binding** | `model.bind_tools([tools])` for function-calling |
| **Agent Creation** | `create_agent()` with built-in tool execution |
| **Structured Output** | Pydantic `BaseModel` with `with_structured_output()` |
| **TypedDict** | Typed dictionaries for state and schema definitions |
| **Message Types** | `SystemMessage`, `HumanMessage`, `AIMessage` |
| **Batch Processing** | Parallel model inference with `model.batch()` |
| **Streaming** | Token-by-token response streaming |
| **Summarization Middleware** | Automatic conversation summarization at trigger thresholds |
| **Checkpointing** | `InMemorySaver` for conversation persistence |
| **Human-in-the-Loop** | Manual approval for tool execution |

#### Key Code (Agent with Tools)

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location"""
    return f"It's sunny in {location}"

model = init_chat_model("openrouter:openai/gpt-oss-120b:free")
agent = create_agent(model=model, tools=[get_weather])

response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Boston?"}]})
```

---

### 3. LangGraph

**Location:** `lang-chain-graph/langgraph.ipynb`

LangGraph extends LangChain by modeling AI workflows as **stateful graphs** with nodes, edges, and conditional routing.

#### Topics Covered

| Concept | Description |
|---------|-------------|
| **StateGraph** | Graph-based execution model with typed state |
| **Nodes** | Computation steps (functions) in the graph |
| **Edges** | Connections between nodes (START → Node → END) |
| **State Management** | `TypedDict` state with `Annotated[list, add_messages]` |
| **Message Accumulation** | `add_messages` reducer for appending to message lists |
| **LLM Chatbot Node** | Graph node wrapping LLM invocation |
| **Conditional Edges** | Branching logic based on state content |
| **ToolNode** | Prebuilt node for tool execution |
| **tools_condition** | Built-in conditional router for tool usage |
| **MemorySaver** | Persistent checkpointing across conversation turns |
| **`thread_id`** | Conversation session isolation |
| **Streaming Modes** | `values`, `updates`, `astream_events` |
| **Graph Visualization** | Mermaid diagram rendering |
| **Human-in-the-Loop** | `interrupt()` for pausing execution for human input |
| **Conditional Routing** | Router function directing to specialized nodes |

#### Architecture Diagram (Agentic LangGraph)

```
START
  │
  ▼
chatbot ──┬── (tools_condition) ──→ ToolNode
  │                                    │
  │                                    │
  └────────────────────────────────────┘
  │
END
```

#### Key Code (Minimal LangGraph Chatbot)

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
result = graph.invoke({"messages": [("human", "Hi")]})
```

#### Key Code (LangGraph with Tools and Conditional Routing)

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

@tool
def ask_human(question: str) -> str:
    """Ask human for help."""
    response = interrupt({"question": question})
    return response["data"]

tools = [ask_human]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=MemorySaver())
```

---

### 4. Guardrails & Safety

**Location:** `gaurdrails-gateway/Gaurdrali_Gateway.ipynb`

Guardrails ensure LLM applications produce **safe, controlled, and compliant outputs** by filtering inputs and outputs at multiple layers.

#### Topics Covered

| Layer | Technique | Description |
|-------|-----------|-------------|
| **Input Guardrails** | Deterministic Keyword Filtering | Block queries containing banned words (hack, exploit, malware, bomb) |
| **Input Guardrails** | Model-based Classification | LLM-as-judge classifies input as SAFE / UNSAFE |
| **Input Guardrails** | Prompt Injection Detection | Regex patterns for jailbreak attempts |
| **Pre-processing** | PII Redaction | Regex-based masking of email, phone, credit card numbers |
| **Orchestration** | LLM Gateway | Multi-provider routing (OpenRouter, Groq, Gemini via LiteLLM) |
| **Agent Middleware** | ContentFilterMiddleware | LangChain middleware intercepting banned keywords before agent execution |
| **Agent Middleware** | HumanInTheLoopMiddleware | Pause agent execution for human approval of sensitive tools |
| **Output Guardrails** | Output Moderation | LLM-as-judge classifies response as SAFE / UNSAFE |

#### Architecture

```
User Input
    │
    ▼
┌─────────────────────┐
│  Input Guardrails    │  ◄── Deterministic + Model-based + Injection Detection
│  • Keyword Filter   │
│  • PII Redaction    │
│  • Injection Check  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LLM Gateway        │  ◄── LiteLLM multi-provider routing
│  • Groq             │
│  • OpenRouter       │
│  • Gemini           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Agent Execution    │  ◄── Human-in-the-loop for sensitive tools
│  • Tool Calling     │
│  • Memory           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Output Guardrails  │  ◄── Model-based output moderation
│  • Safety Check     │
└─────────┬───────────┘
          │
          ▼
    Final Response
```

#### Key Code (PII Redaction)

```python
import re

PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"(\+91[\-\s]?)?[6-9]\d{9}",
    "CREDIT_CARD": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
}

def redact_pii(text: str) -> str:
    clean = text
    for label, pattern in PII_PATTERNS.items():
        clean = re.sub(pattern, f"<{label}_REDACTED>", clean)
    return clean
```

#### Key Code (Model-based Input Guardrail)

```python
from litellm import completion

def model_guardrail(text: str) -> str:
    response = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"Classify this as SAFE or UNSAFE.\n\nInput:\n{text}\n\nReply with ONLY:\nSAFE\nor\nUNSAFE"
        }],
        max_tokens=5
    )
    return response.choices[0].message.content.strip()
```

---

### 5. Model Context Protocol (MCP)

**Location:** `mcp/` directory

MCP is a standardized protocol for exposing tools and resources from external servers to LLM applications. This implementation uses `FastMCP` for the server and `langchain-mcp-adapters` for the client.

#### Server (`mcp_server.py`)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b

if __name__ == "__main__":
    mcp.run()
```

#### Client (`main.py`)

```python
import asyncio
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio"
        }
    })

    tools = await client.get_tools()
    llm = init_chat_model("openrouter:openai/gpt-oss-120b:free")
    agent = create_react_agent(llm, tools)

    response = await agent.ainvoke({
        "messages": [("user", "What is 25 multiplied by 4?")]
    })
    print(response["messages"][-1].content)

asyncio.run(main())
```

#### How It Works

1. **MCP Server** registers tools (`add`, `multiply`) via decorators
2. **MCP Client** connects to the server via stdio transport
3. **LangChain Adapter** (`MultiServerMCPClient`) converts MCP tools into LangChain-compatible tools
4. **ReAct Agent** uses the tools autonomously — the LLM decides when to call `multiply`, extracts arguments, and receives results

---

### 6. Tool Calling

**Location:** `sample/tool-callin-sample.py`

A from-scratch implementation of tool calling — the fundamental pattern that enables LLMs to interact with external functions.

#### How It Works

1. The LLM receives a system prompt instructing it to output **JSON** for tool calls
2. When the model returns `{"tool": "tool_name", "arguments": {...}}`, the client executes the function
3. The result is fed back to the model for further reasoning
4. When satisfied, the model returns `{"final": "answer"}`

#### Architecture

```
User: "What's the weather in Delhi?"
    │
    ▼
LLM → {"tool": "get_weather", "arguments": {"city": "Delhi"}}
    │
    ▼
execute_tool("get_weather", {"city": "Delhi"}) → "Weather in Delhi: 34C"
    │
    ▼
LLM (with tool result) → {"final": "The weather in Delhi is 34°C and sunny."}
    │
    ▼
AI: "The weather in Delhi is 34°C and sunny."
```

#### Key Code

```python
TOOLS = {"get_weather": get_weather}

def execute_tool(tool_name, arguments):
    if tool_name not in TOOLS:
        raise Exception("Invalid tool")
    return TOOLS[tool_name](**arguments)

# Inside the conversation loop:
parsed = json.loads(content)
if "tool" in parsed:
    tool_result = execute_tool(parsed["tool"], parsed["arguments"])
    messages.append({"role": "tool", "content": str(tool_result)})
    continue  # Let the model process the result
if "final" in parsed:
    print(f"AI: {parsed['final']}")
    break
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/MishraShardendu22/Langchain-Learning.git
cd Gen-AI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# For Jupyter notebooks
pip install jupyter
jupyter notebook
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPEN_ROUTER=your_openrouter_api_key
GROQ=your_groq_api_key
GEMINI=your_gemini_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key  # Optional, for LangSmith tracing
```

## Dependencies

**`pyproject.toml`** defines the project. Key packages used across the notebooks:

| Package | Purpose |
|---------|---------|
| `langchain` | Core LLM framework |
| `langchain-openai` | OpenAI / OpenRouter integration |
| `langchain-google-genai` | Gemini model integration |
| `langchain-groq` | Groq model integration |
| `langchain-community` | Community integrations (FAISS, BM25, etc.) |
| `langchain-mcp-adapters` | MCP protocol client adapter |
| `langgraph` | Graph-based agent workflows |
| `langsmith` | LLM application evaluation & tracing |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Embedding models |
| `pymupdf` | PDF document loading |
| `litellm` | Multi-provider LLM gateway |
| `python-dotenv` | Environment variable management |
| `openai` | OpenAI API client |
| `cohere` | Cohere rerank API |
| `mcp` (FastMCP) | Model Context Protocol server |

---

## License

This project is for **learning and educational purposes**.

---

*Built with curiosity. One notebook at a time.*