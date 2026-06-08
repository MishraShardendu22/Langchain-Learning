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