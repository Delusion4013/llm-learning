from fastmcp import FastMCP

mcp = FastMCP(name="simple_mcp_server", port=8000)

@mcp.tool
def hello(name: str) -> str:
    """
    A simple tool that greets the user by name.
    """
    return f"Hello from MCP, {name}!"

mcp.run()
# This will start the MCP server and make the `hello` tool available for use.