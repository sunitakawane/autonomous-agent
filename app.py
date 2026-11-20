"""
Simple UI for the autonomous email agent.
Run
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""

import gradio as gr

from agent import MODEL, client, run_email_agent

SAMPLE_EMAIL = """
Hi, my order #A1042 was supposed to arrive yesterday, but it hasn't arrived.
Can you please tell me what's happening? I need it by Friday.
"""

SAMPLE_EMAIL_UNKNOWN_ORDER = """
Hi, my order #A9999 was supposed to arrive yesterday, but it hasn't arrived.
Can you please tell me what's happening?
"""


def handle_run(email: str):
    """Run the agent and show its step-by-step trace, then the final draft."""
    if not email.strip():
        return "Please paste a customer email first.", ""

    log = []
    try:
        draft = run_email_agent(email, log=log)
    except Exception as exc:
        return f"Error: {exc}", ""

    return "\n".join(log), draft


with gr.Blocks(title="Autonomous Email Agent") as demo:
    gr.Markdown(
        f"""# Autonomous Email Agent
        **Agent = Model + Harness** — the model decides what to do next,
        the harness runs the loop and executes tools.

        Model: `{MODEL}` · Base URL: `{client.base_url}`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            email_box = gr.Textbox(
                label="Customer email",
                value=SAMPLE_EMAIL,
                lines=6,
            )
            run_btn = gr.Button("Run agent", variant="primary")
            gr.Examples(
                examples=[
                    [SAMPLE_EMAIL],
                    [SAMPLE_EMAIL_UNKNOWN_ORDER],
                ],
                inputs=email_box,
                label="Try an example",
            )

        with gr.Column(scale=1):
            trace_box = gr.Textbox(
                label="Agent trace (reason → act → observe)",
                lines=16,
                interactive=False,
            )
            draft_box = gr.Textbox(
                label="Final email draft",
                lines=10,
                interactive=False,
            )

    run_btn.click(fn=handle_run, inputs=email_box, outputs=[trace_box, draft_box])

if __name__ == "__main__":
    demo.launch()