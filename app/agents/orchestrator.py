"""
LangGraph Multi-Agent Orchestrator.

Orchestrates user query intent classification, RAG retrieval, memory recall,
routing to specialized agent nodes, passive safety verification, memory storage,
and final response packaging.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes.protocol_agent import protocol_agent_node
from app.agents.nodes.tracking_agent import tracking_agent_node
from app.agents.nodes.safety_agent import safety_agent_node
from app.agents.nodes.troubleshooting_agent import troubleshooting_agent_node
from app.agents.nodes.research_agent import research_agent_node
from app.agents.nodes.recommendation_agent import recommendation_agent_node
from app.agents.nodes.memory_agent import memory_recall_node, memory_save_node
from app.core.llm import get_fast_llm
from app.retrieval.pipeline import RAGPipeline
from app.models.enums import QueryType

logger = logging.getLogger(__name__)

# Single global RAG pipeline instance
_rag_pipeline = RAGPipeline()


# ── Node Definitions ─────────────────────────────────────────────────


async def classify_node(state: AgentState) -> dict:
    """Classifies the user query intent to route to the correct agent node."""
    logger.info("Classifying query: %s", state["user_query"][:80])
    
    try:
        llm = get_fast_llm()
        system_prompt = (
            "You are a laboratory conversation classifier. Classify the user query into exactly one of these categories:\n"
            "- 'protocol_question': User is asking about steps, parameters, timing, or details of the active protocol.\n"
            "- 'experiment_update': User is reporting step progress, completion, or actions they performed (e.g. 'I finished step 3').\n"
            "- 'safety_concern': User is asking about hazards, PPE, chemical compatibility, or disposal safety.\n"
            "- 'troubleshooting': User is reporting a failure, unexpected result, or problem in the lab (e.g. 'gel was blank').\n"
            "- 'conceptual_question': User is asking general biology/chemistry explanation questions (e.g. 'how does PCR work?').\n"
            "- 'protocol_request': User is asking to recommend or find protocols (e.g. 'suggest a miniprep protocol').\n"
            "- 'general_chat': General greetings or social conversation.\n"
            "\n"
            "Respond with only the classification name string. Do not include quotes or surrounding text."
        )

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ])

        classification = response.content.strip().lower()
        # Normalize classification to match enum values
        valid_types = {
            "protocol_question": QueryType.PROTOCOL_QUESTION.value,
            "experiment_update": QueryType.EXPERIMENT_UPDATE.value,
            "safety_concern": QueryType.SAFETY_CONCERN.value,
            "troubleshooting": QueryType.TROUBLESHOOTING.value,
            "conceptual_question": QueryType.CONCEPTUAL_QUESTION.value,
            "protocol_request": QueryType.PROTOCOL_REQUEST.value,
            "general_chat": QueryType.GENERAL_CHAT.value,
        }

        query_type = valid_types.get(classification, QueryType.GENERAL_CHAT.value)
        logger.info("Query classified as: %s", query_type)
        return {"query_type": query_type}
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return {"query_type": QueryType.GENERAL_CHAT.value}


async def retrieve_node(state: AgentState) -> dict:
    """Runs the RAG pipeline to retrieve context relevant to the user query."""
    query_type = state.get("query_type")
    
    # We skip retrieval for general chat
    if query_type == QueryType.GENERAL_CHAT.value:
        return {"retrieved_context": [], "sources": []}

    logger.info("Retrieving RAG context for query: %s", state["user_query"][:80])

    try:
        # Determine Pinecone namespace
        namespace = "protocols"
        if state.get("current_protocol") and state["current_protocol"].get("pinecone_namespace"):
            namespace = state["current_protocol"]["pinecone_namespace"]

        result = await _rag_pipeline.retrieve(
            query=state["user_query"],
            namespace=namespace,
        )

        retrieved_context = [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in result.chunks
        ]

        return {
            "retrieved_context": retrieved_context,
            "sources": result.citations,
        }
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return {"retrieved_context": [], "sources": []}


async def safety_check_node(state: AgentState) -> dict:
    """
    Passive safety check on the agent response.

    Looks for any chemical hazard mentions, missing safety instructions, or PPE omissions.
    """
    response_text = state.get("response")
    if not response_text:
        return {}

    logger.info("Performing passive safety check on generated response")

    try:
        llm = get_fast_llm()
        system_prompt = (
            "You are a laboratory safety validator. Analyze the assistant response and determine if any safety warning or PPE reminder is missing, "
            "or if the response advises doing anything unsafe.\n"
            "\n"
            "Format your response as a JSON object with keys:\n"
            "- 'safety_alert': true or false\n"
            "- 'alert_message': string warning or null if false\n"
            "- 'hazard_level': 'low', 'medium', 'high', or 'critical' (null if safety_alert is false)\n"
            "\n"
            "Do NOT return conversational text. Return ONLY the JSON object."
        )

        analysis = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Assistant Response:\n{response_text}"),
        ])

        import json
        analysis_text = analysis.content.strip()
        if analysis_text.startswith("```json"):
            analysis_text = analysis_text[7:]
        if analysis_text.endswith("```"):
            analysis_text = analysis_text[:-3]

        data = json.loads(analysis_text.strip())

        if data.get("safety_alert") and data.get("alert_message"):
            alert = {
                "message": data["alert_message"],
                "hazard_level": data.get("hazard_level", "medium"),
                "source": "passive_safety_check",
            }
            logger.warning("Passive safety alert triggered: %s", data["alert_message"])
            return {"safety_alerts": [alert]}
    except Exception as e:
        logger.error("Passive safety check failed: %s", e)

    return {}


async def respond_node(state: AgentState) -> dict:
    """包装最终响应，添加安全警报和来源引用。"""
    response = state.get("response", "I'm sorry, I couldn't generate a response.")
    safety_alerts = state.get("safety_alerts", [])

    # Append safety alerts to response if they exist and are not already mentioned
    if safety_alerts:
        alert_text = "\n\n⚠️ **Safety Warnings:**"
        for alert in safety_alerts:
            alert_text += f"\n- [{alert.get('hazard_level', 'warning').upper()}] {alert.get('message')}"
        response += alert_text

    return {"response": response}


# ── Edge Routing ─────────────────────────────────────────────────────


def route_to_agent(state: AgentState) -> str:
    """Routes the state to the correct agent node based on classified query_type."""
    q_type = state.get("query_type")
    
    if q_type == QueryType.PROTOCOL_QUESTION.value:
        return "protocol_agent"
    elif q_type == QueryType.EXPERIMENT_UPDATE.value:
        return "tracking_agent"
    elif q_type == QueryType.SAFETY_CONCERN.value:
        return "safety_agent"
    elif q_type == QueryType.TROUBLESHOOTING.value:
        return "troubleshooting_agent"
    elif q_type == QueryType.CONCEPTUAL_QUESTION.value:
        return "research_agent"
    elif q_type == QueryType.PROTOCOL_REQUEST.value:
        return "recommendation_agent"
    else:
        # Default fallback for general chat or unclassified queries
        return "research_agent"


# ── Graph Builder ────────────────────────────────────────────────────


def build_graph(checkpointer: Any = None):
    """
    Builds and compiles the LangGraph StateGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("memory_recall", memory_recall_node)
    
    workflow.add_node("protocol_agent", protocol_agent_node)
    workflow.add_node("tracking_agent", tracking_agent_node)
    workflow.add_node("safety_agent", safety_agent_node)
    workflow.add_node("troubleshooting_agent", troubleshooting_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("recommendation_agent", recommendation_agent_node)

    workflow.add_node("safety_check", safety_check_node)
    workflow.add_node("memory_save", memory_save_node)
    workflow.add_node("respond", respond_node)

    # Define DAG Flow
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "memory_recall")

    # Intent routing to specialized agents
    workflow.add_conditional_edges(
        "memory_recall",
        route_to_agent,
        {
            "protocol_agent": "protocol_agent",
            "tracking_agent": "tracking_agent",
            "safety_agent": "safety_agent",
            "troubleshooting_agent": "troubleshooting_agent",
            "research_agent": "research_agent",
            "recommendation_agent": "recommendation_agent",
        }
    )

    # Gather responses into safety check
    for agent in [
        "protocol_agent",
        "tracking_agent",
        "safety_agent",
        "troubleshooting_agent",
        "research_agent",
        "recommendation_agent",
    ]:
        workflow.add_edge(agent, "safety_check")

    workflow.add_edge("safety_check", "memory_save")
    workflow.add_edge("memory_save", "respond")
    workflow.add_edge("respond", END)

    # Compile
    return workflow.compile(checkpointer=checkpointer)


def get_orchestrator():
    """Returns a compiled instance of the state graph with a memory saver checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver
    return build_graph(checkpointer=MemorySaver())
