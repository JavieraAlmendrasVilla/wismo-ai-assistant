"""
gradio_ui.py
------------
Two-tab Gradio chat interface for the WISMO AI Assistant.

Tab 1 — "Track My Parcel": structured lookup with raw data panel
Tab 2 — "Ask WISMO"       : free-text multi-turn chat via LangGraph agent

Run:
    python app/gradio_ui.py
    → opens at http://localhost:7860
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

import gradio as gr

from agent.wismo_chain import build_wismo_chain
from agent.wismo_agent import chat as agent_chat
from guardrails.output_validator import OutputValidator
from guardrails.pii_filter import PIIFilter

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_chain = None
_validator = OutputValidator()
_pii = PIIFilter()


def _get_chain():
    global _chain
    if _chain is None:
        _chain = build_wismo_chain()
    return _chain


# ---------------------------------------------------------------------------
# Tab 1: Track My Parcel
# ---------------------------------------------------------------------------

def track_parcel(tracking_id: str) -> tuple[str, str, str]:
    """
    Look up a tracking ID and return:
      (ai_summary, guardrail_status_html, raw_data_json)
    """
    if not tracking_id or not tracking_id.strip():
        return (
            "Please enter a tracking ID.",
            _guardrail_badge(None),
            "",
        )

    tracking_id = tracking_id.strip()

    try:
        result = _get_chain().invoke({
            "tracking_id": tracking_id,
            "query": f"What is the current status of tracking ID {tracking_id}?",
        })
    except Exception as exc:
        logger.error("Chain error: %s", exc)
        return (
            f"An error occurred while looking up your parcel. Please try again.\n\nDetail: {exc}",
            _guardrail_badge(False),
            "",
        )

    # Run guardrail
    validation = _validator.validate(result.answer, result.tracking_data)
    final_answer = validation.safe_response

    # Build display-safe raw data
    safe_data = _pii.safe_to_display(result.tracking_data)
    raw_json = _format_raw_data(safe_data) if safe_data else "No tracking data found."

    return final_answer, _guardrail_badge(validation.is_valid), raw_json


def _guardrail_badge(is_valid: bool | None) -> str:
    """Return HTML for the guardrail status badge."""
    if is_valid is None:
        return '<span style="color:#888">⬤ Guardrail: N/A</span>'
    if is_valid:
        return '<span style="color:#22c55e; font-weight:bold">✅ Guardrail: PASSED</span>'
    return '<span style="color:#ef4444; font-weight:bold">🛡️ Guardrail: TRIGGERED — response sanitised</span>'


def _format_raw_data(data: dict) -> str:
    """Format tracking dict as readable text."""
    if not data:
        return "No data."
    lines = []
    for k, v in data.items():
        if v is not None:
            label = k.replace("_", " ").title()
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2: Ask WISMO (multi-turn agent)
# ---------------------------------------------------------------------------

def agent_respond(
    user_message: str,
    chat_history: list[dict],
) -> tuple[list[dict], str]:
    """
    Process a user message through the LangGraph agent and update chat history.

    Returns:
        (updated_chat_history, guardrail_html)
    """
    if not user_message or not user_message.strip():
        return chat_history, _guardrail_badge(None)

    # Convert gr.ChatInterface history format → internal format
    history_dicts = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history
    ]

    try:
        response = agent_chat(user_message.strip(), history=history_dicts)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        response = (
            "I'm sorry, I encountered an error processing your request. "
            "Please try again or contact DHL support."
        )

    # Run guardrail on agent response (no retrieved_data available at this level)
    validation = _validator.validate(response, {})
    final_response = validation.safe_response

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_response},
    ]

    return chat_history, _guardrail_badge(validation.is_valid)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
.guardrail-box { font-size: 0.95em; padding: 6px 10px; border-radius: 6px; }
.raw-data-box { font-family: monospace; font-size: 0.85em; }
footer { display: none !important; }
"""

_EXAMPLE_IDS = [
    "JD014455669988112233",
    "GM987654321098765432",
    "LX112233445566778899",
]

_EXAMPLE_QUERIES = [
    "Where is my package JD014455669988112233?",
    "My order seems delayed, tracking GM987654321098765432",
    "Has my parcel LX112233445566778899 been delivered?",
    "What's wrong with my shipment? I ordered last week.",
    "Track my order",
]


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    with gr.Blocks(
        title="DHL WISMO AI Assistant",
        css=_CSS,
        theme=gr.themes.Soft(primary_hue="yellow"),
    ) as demo:

        gr.Markdown(
            """
            # DHL WISMO AI Assistant
            ### *Where Is My Order? — Powered by RAG + LangGraph*
            """
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1: Track My Parcel
            # ------------------------------------------------------------------
            with gr.Tab("Track My Parcel"):
                gr.Markdown("Enter your DHL tracking number to get an instant AI-powered status update.")

                with gr.Row():
                    tid_input = gr.Textbox(
                        label="Tracking Number",
                        placeholder="e.g. JD014455669988112233",
                        scale=4,
                    )
                    track_btn = gr.Button("Track", variant="primary", scale=1)

                guardrail_html_1 = gr.HTML(
                    value=_guardrail_badge(None),
                    label="Guardrail Status",
                    elem_classes=["guardrail-box"],
                )

                ai_output = gr.Textbox(
                    label="AI Summary",
                    lines=4,
                    interactive=False,
                )

                with gr.Accordion("Raw Tracking Data (for transparency)", open=False):
                    raw_data_output = gr.Textbox(
                        label="Retrieved Data",
                        lines=10,
                        interactive=False,
                        elem_classes=["raw-data-box"],
                    )

                gr.Examples(
                    examples=[[eid] for eid in _EXAMPLE_IDS],
                    inputs=[tid_input],
                    label="Example Tracking IDs",
                )

                track_btn.click(
                    fn=track_parcel,
                    inputs=[tid_input],
                    outputs=[ai_output, guardrail_html_1, raw_data_output],
                )
                tid_input.submit(
                    fn=track_parcel,
                    inputs=[tid_input],
                    outputs=[ai_output, guardrail_html_1, raw_data_output],
                )

            # ------------------------------------------------------------------
            # Tab 2: Ask WISMO
            # ------------------------------------------------------------------
            with gr.Tab("Ask WISMO"):
                gr.Markdown(
                    "Chat freely with the WISMO agent. It can look up tracking IDs, "
                    "estimate delivery dates, and find similar cases."
                )

                chatbot = gr.Chatbot(
                    label="WISMO Assistant",
                    height=420,
                    type="messages",
                    avatar_images=(None, "https://www.dhl.com/favicon.ico"),
                )

                guardrail_html_2 = gr.HTML(
                    value=_guardrail_badge(None),
                    elem_classes=["guardrail-box"],
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask anything about your DHL shipment…",
                        label="Your message",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                clear_btn = gr.Button("Clear conversation", size="sm")

                gr.Examples(
                    examples=[[q] for q in _EXAMPLE_QUERIES],
                    inputs=[msg_input],
                    label="Example questions",
                )

                def submit_message(message, history):
                    history, badge = agent_respond(message, history)
                    return history, badge, ""

                send_btn.click(
                    fn=submit_message,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, guardrail_html_2, msg_input],
                )
                msg_input.submit(
                    fn=submit_message,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, guardrail_html_2, msg_input],
                )
                clear_btn.click(
                    fn=lambda: ([], _guardrail_badge(None)),
                    outputs=[chatbot, guardrail_html_2],
                )

        gr.Markdown(
            "<small>WISMO AI Assistant — Demo project for DHL logistics. "
            "All data is synthetic. Not for production use.</small>"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
