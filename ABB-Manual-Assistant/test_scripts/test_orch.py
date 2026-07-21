# generated test file - orchestrator agent using search tool

import os
import sys


# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import logging

####
import warnings

from orchestrator_agent import Orchestrator


# Suppress all warnings (UserWarning, DeprecationWarning, etc.)
warnings.filterwarnings("ignore")
# Suppress log messages from all libraries
logging.basicConfig(level=logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
####


async def main():
    orchestrator = Orchestrator()
    response = await orchestrator.run("How to integrate IRC5?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
