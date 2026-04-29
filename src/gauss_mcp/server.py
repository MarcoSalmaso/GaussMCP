from gauss_mcp.app import mcp

# Import all tool modules so their @mcp.tool() decorators fire and register every tool.
import gauss_mcp.tools.data_loader  # noqa: F401
import gauss_mcp.tools.descriptive  # noqa: F401
import gauss_mcp.tools.inferential  # noqa: F401
import gauss_mcp.tools.time_series  # noqa: F401
import gauss_mcp.tools.bayesian     # noqa: F401


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
