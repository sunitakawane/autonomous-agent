"""
Simple autonomous email agent
Model + harness = Agent

"""

from openai import OpenAI
import os
from dotenv import load_dotenv
import json
# ---------------------------------------------------------------------------
# 1. Set up the model
# ---------------------------------------------------------------------------
load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY", "none")
MODEL = os.getenv("LLM_MODEL", "Qwen3.8-Flash-Next")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# ---------------------------------------------------------------------------
# 2. The customer email and a tiny mock order database
# ---------------------------------------------------------------------------
customer_email = """
Hi, my order #A1042 was supposed to arrive yesterday, but it hasn't arrived.
Can you please tell me what's happening? I need it by Friday.
"""


orders = {
    "A1042": {
        "customer_name": "Sam",
        "status": "In transit",
        "last_update": "Package departed the Dallas sorting facility this morning",
        "estimated_delivery": "Thursday",
        "carrier": "Example Express"
    }
}


# ---------------------------------------------------------------------------
# 3. Give the agent tools (read-only: it can prepare a reply, nothing else)
# ---------------------------------------------------------------------------
def lookup_order(order_id: str) -> dict:
    """Look up the latest available information for an order."""
    order = orders.get(order_id)

    if not order:
        return {
            "found": False,
            "message": f"No order was found for {order_id}."
        }

    return {
        "found": True,
        "order_id": order_id,
        **order
    }



# Describe the tool to the model using the tool-calling schema.
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up the current status and estimated delivery date for a customer order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID from the customer's email, for example A1042."
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]

available_functions = {
    "lookup_order": lookup_order
}

# ---------------------------------------------------------------------------
# 4. Define the agent's goal and boundaries
# ---------------------------------------------------------------------------

system_prompt = """
You are a customer-support email assistant.

Your goal is to prepare an accurate, helpful draft reply to the customer's email.

You may use the available tools when you need information.
Decide what to do based on the email and the information you have.
After observing a tool result, decide whether you need another action or can finish.

Rules:
- Never invent order details, delivery dates, or policies.
- If the order cannot be found, say that the information is unavailable and ask for clarification.
- Do not claim that an email has been sent. You may only draft a reply.
- Do not promise delivery by a date unless the available order information supports it.
- Keep the reply polite, concise, and specific.
- When you have enough information, provide the final email draft.
"""


# ---------------------------------------------------------------------------
# 5. Build the autonomous loop (the harness)
# ---------------------------------------------------------------------------
def run_email_agent(email: str, max_steps: int = 5, log: list | None = None) -> str:
    """
    Run the agent loop on a customer email.

    If `log` (a list) is provided, each step is appended to it as it happens.
    This lets the UI show the agent's progress live.
    """
    def emit(line: str):
        if log is not None:
            log.append(line)
        print(line)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please handle this customer email:\n{email}"}
    ]

    for step in range(1, max_steps + 1):
        emit(f"\n--- Agent step {step} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2
        )

        assistant_message = response.choices[0].message

        # Keep the assistant's tool-call request in the conversation history.
        messages.append(assistant_message.model_dump(exclude_none=True))

        # If the model did not request a tool, it has produced its final response.
        if not assistant_message.tool_calls:
            emit("Agent finished.")
            return assistant_message.content

        # Execute each tool requested by the model.
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            emit(f"Action selected: {function_name}({arguments})")

            function_to_call = available_functions.get(function_name)
            if function_to_call is None:
                tool_result = {"error": f"Tool {function_name} is not available."}
            else:
                try:
                    tool_result = function_to_call(**arguments)
                except Exception as exc:
                    tool_result = {"error": str(exc)}

            emit(f"Observation: {tool_result}")

            # Feed the observation back to the model.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

    return (
        "The agent reached its step limit without producing a final draft. "
        "A human should review the task."
    )


# ---------------------------------------------------------------------------
# Run directly from the terminal: python agent.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model:  {MODEL}")
    print(f"Base:   {client.base_url}\n")
    print(customer_email)

    draft = run_email_agent(customer_email)
    print("\n===== FINAL EMAIL DRAFT =====\n")
    print(draft)