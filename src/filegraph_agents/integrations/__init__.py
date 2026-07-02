"""
filegraph_agents.integrations — Adapters for external platforms.

Each submodule wraps FileGraph Agents (FGA) for a specific grading /
infrastructure system.  Import the desired adapter class or constant
directly from this package:

    from filegraph_agents.integrations import FGAAgent, FGA_GIT_URL, REPO_DIR
"""

from filegraph_agents.integrations.pier import FGAAgent, FGA_GIT_URL, REPO_DIR

__all__ = [
    "FGAAgent",
    "FGA_GIT_URL",
    "REPO_DIR",
]
