"""
LangGraph Multi-Agent System for the AI Research Protocol Assistant.

Provides specialized agents for protocol understanding, experiment tracking,
safety monitoring, troubleshooting, research Q&A, and recommendations —
all orchestrated via a LangGraph StateGraph.
"""

from app.agents.state import AgentState
from app.agents.orchestrator import build_graph

__all__ = ["AgentState", "build_graph"]
