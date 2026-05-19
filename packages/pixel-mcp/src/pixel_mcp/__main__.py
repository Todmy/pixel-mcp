"""Allow `python -m pixel_mcp` to invoke the CLI."""

from pixel_mcp.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
