"""Allow `python -m pixel_mcp_ml` to invoke the CLI."""

from pixel_mcp_ml.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
