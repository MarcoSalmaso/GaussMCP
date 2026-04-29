from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "GaussMCP",
    instructions=(
        "Statistical analysis MCP server. "
        "Provides tools for descriptive statistics, inferential tests, "
        "time series analysis, and Bayesian inference. "
        "Supports data from CSV, Excel, PostgreSQL, and inline JSON."
    ),
)
