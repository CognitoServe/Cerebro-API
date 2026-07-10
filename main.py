"""
main.py — CLI entry point for the async research agent.
Lets you run a single question without starting the FastAPI server.

Usage:
    python main.py "What are the performance improvements in Python 3.13?"
"""
from __future__ import annotations

import asyncio
import json
import sys

from dotenv import load_dotenv

from agent.report_schema import AgentFailure, ResearchReport
from agent.runner        import run_agent


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your research question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result   = asyncio.run(run_agent(question))

    if isinstance(result, ResearchReport):
        print("\n" + "=" * 72)
        print("RESEARCH REPORT")
        print("=" * 72)
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    elif isinstance(result, AgentFailure):
        print(f"\n[FAILURE] reason={result.reason}\n{result.detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
