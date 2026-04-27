"""
wismo_agent.py
--------------
LangGraph ReAct agent for multi-turn WISMO queries with memory and tools.

Tools:
  - query_tracking_db     : structured lookup by tracking_id
  - get_delivery_estimate : calculates / surfaces ETA from DB
  - find_similar_cases    : semantic search for similar shipment issues

State: {messages, tracking_id, retrieved_data, response, error}
"""

import os
import logging
from datetime import date
from typing import Annotated, Any, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from retrieval.structured_retriever import get_tracking_by_id
from retrieval.semantic_retriever import semantic_search

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Mutable state threaded through all graph nodes."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tracking_id: Optional[str]
    retrieved_data: Optional[dict]
    response: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def query_tracking_db(tracking_id: str) -> str:
    """
    Look up a DHL shipment by tracking ID and return its current status.

    Args:
        tracking_id: The DHL tracking ID (e.g. JD012345678901234567).

    Returns:
        Human-readable shipment status string, or a not-found message.
    """
    data = get_tracking_by_id(tracking_id.strip())
    if data is None:
        return f"No shipment found for tracking ID '{tracking_id}'. Please verify the ID and try again."

    return (
        f"Tracking ID: {data['tracking_id']}\n"
        f"Status: {data['status_code']}\n"
        f"Current Location: {data['current_location']}\n"
        f"Destination: {data['destination_city']}\n"
        f"Estimated Delivery: {data['estimated_delivery']}\n"
        f"Last Update: {data['last_update']}\n"
        f"Description: {data['status_description']}"
    )


@tool
def get_delivery_estimate(tracking_id: str) -> str:
    """
    Return the estimated (or actual) delivery date for a shipment.

    Args:
        tracking_id: The DHL tracking ID.

    Returns:
        Delivery estimate string with context about delays.
    """
    data = get_tracking_by_id(tracking_id.strip())
    if data is None:
        return f"Cannot determine delivery estimate — tracking ID '{tracking_id}' not found."

    if data.get("actual_delivery"):
        return f"Package was delivered on {data['actual_delivery']}."

    estimated = data.get("estimated_delivery", "unknown")
    status = data.get("status_code", "")

    if status == "DELAYED":
        return (
            f"Estimated delivery date is {estimated}. "
            "Note: This shipment is currently delayed — the estimate may change."
        )
    if status == "EXCEPTION":
        return (
            f"Estimated delivery date is {estimated}. "
            "However, a delivery exception has been raised. Please contact DHL support."
        )

    today = date.today().isoformat()
    if estimated < today and status not in ("DELIVERED",):
        return (
            f"Estimated delivery was {estimated} (overdue). "
            "Current status: {status}. Please contact DHL for an update."
        ).format(status=status, estimated=estimated)

    return f"Estimated delivery date: {estimated}. Current status: {status}."


@tool
def find_similar_cases(issue_description: str) -> str:
    """
    Search for shipments with similar status descriptions to help diagnose issues.

    Args:
        issue_description: Natural language description of the problem
            (e.g. "package stuck at customs", "failed delivery attempt").

    Returns:
        Formatted list of similar cases from the database.
    """
    results = semantic_search(issue_description, top_k=3)
    if not results:
        return "No similar cases found in the database."

    lines = ["Similar cases found:\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['status_code']}] {r['status_description']} "
            f"(Tracking: {r['tracking_id']}, Location: {r['current_location']})"
        )
    return "\n".join(lines)


TOOLS = [query_tracking_db, get_delivery_estimate, find_similar_cases]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm_with_tools():
    """Build the LLM instance bound with the agent tools."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("MODEL_NAME", "llama3.2")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        return llm.bind_tools(TOOLS)

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'.")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM = """\
You are the DHL WISMO (Where Is My Order?) AI assistant.

You have access to three tools:
- query_tracking_db: look up any shipment by tracking ID
- get_delivery_estimate: get the ETA for a shipment
- find_similar_cases: find shipments with similar issues for context

RULES:
1. Always use the tools — do not answer from memory or make up information.
2. If the user provides a tracking ID, always call query_tracking_db first.
3. Never reveal, guess, or fabricate addresses, names, or delivery dates.
4. If asked to do anything outside of DHL tracking (legal advice, refunds, etc.),
   politely decline and direct the user to DHL customer service.
5. Keep responses professional, empathetic, and concise.
"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def agent_node(state: AgentState, llm_with_tools) -> AgentState:
    """Call the LLM; it decides whether to use a tool or respond directly."""
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM)] + messages

    response = llm_with_tools.invoke(messages)
    return {**state, "messages": [response]}


def not_found_node(state: AgentState) -> AgentState:
    """Fallback node when no tracking data is found after tool calls."""
    fallback = (
        "I'm sorry, I couldn't find any tracking information for the ID you provided. "
        "Please double-check the tracking number (it usually starts with 'JD', 'GM', 'LX', or 'RX' "
        "followed by 18 digits) or contact DHL customer support at 1-800-225-5345."
    )
    return {
        **state,
        "response": fallback,
        "messages": [AIMessage(content=fallback)],
        "error": "tracking_id_not_found",
    }


def should_continue(state: AgentState) -> str:
    """Router: decide whether to call a tool, go to not_found, or end."""
    last_message = state["messages"][-1]

    # If the LLM produced a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Check if one of the tool results was a not-found signal
        return "tools"

    # No tool call → we have a final answer
    return END


def after_tools(state: AgentState) -> str:
    """
    After running tools: check if all tool results indicate not-found,
    then route to not_found_node; otherwise continue to agent.
    """
    messages = state["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if tool_messages:
        last_tool_msg = tool_messages[-1]
        if "not found" in last_tool_msg.content.lower() and len(tool_messages) >= 2:
            return "not_found"

    return "agent"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_wismo_agent():
    """
    Build and compile the LangGraph ReAct agent.

    Returns:
        A compiled LangGraph app (callable with .invoke() or .stream()).
    """
    llm_with_tools = _build_llm_with_tools()
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", lambda s: agent_node(s, llm_with_tools))
    graph.add_node("tools", tool_node)
    graph.add_node("not_found", not_found_node)

    # Edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "not_found": "not_found"})
    graph.add_edge("not_found", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    """Return a cached WISMO agent instance."""
    global _agent
    if _agent is None:
        _agent = build_wismo_agent()
    return _agent


def chat(user_message: str, history: Optional[list[dict]] = None) -> str:
    """
    High-level helper for single-turn or multi-turn chat.

    Args:
        user_message: The user's latest message.
        history: Optional list of prior {role, content} dicts for multi-turn.

    Returns:
        Agent's text response.
    """
    agent = get_agent()

    messages: list[BaseMessage] = []
    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=user_message))

    initial_state: AgentState = {
        "messages": messages,
        "tracking_id": None,
        "retrieved_data": None,
        "response": None,
        "error": None,
    }

    result = agent.invoke(initial_state)
    final_messages = result.get("messages", [])
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return result.get("response", "I'm sorry, I was unable to process your request.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from retrieval.structured_retriever import list_all_tracking_ids

    logging.basicConfig(level="INFO")
    ids = list_all_tracking_ids()
    tid = ids[0] if ids else "UNKNOWN"
    print(f"Testing agent with: 'Where is my package {tid}?'")
    answer = chat(f"Where is my package {tid}?")
    print("\nAgent:", answer)
