import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types


load_dotenv(dotenv_path=Path(__file__).parent / ".env")
client = genai.Client()


LOG_FILE = Path(__file__).parent / "day4_validation_failures.jsonl"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_validation_failure(tool_name: str, raw_args: dict, rule_broken: str, context: str) -> None:

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": tool_name,
        "arguments": raw_args,
        "rule_broken": rule_broken,
        "context": context,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logging.warning(f"[GUARDRAIL BLOCKED] {tool_name} -> {rule_broken}")


# 3. Simulated Source-of-Truth Database
ORDERS_DB = {
    "ORD-101": {
        "item": "Mechanical Keyboard",
        "amount": 4500.0,
        "days_ago": 5,
        "status": "DELIVERED",
    },
    "ORD-102": {
        "item": "USB-C Hub",
        "amount": 1200.0,
        "days_ago": 45,  # Outside 30-day return policy!
        "status": "DELIVERED",
    },
    "ORD-103": {
        "item": "Laptop Stand",
        "amount": 1800.0,
        "days_ago": 1,
        "status": "IN_TRANSIT",  # Not delivered yet!
    },
}



class RefundRequest(BaseModel):
    order_id: str = Field(description="Order ID in format ORD-XXX")
    refund_amount: float = Field(gt=0, description="Positive refund amount in INR")
    reason: str = Field(min_length=5, description="Brief customer reason for refund")


def lookup_order(order_id: str) -> dict:
    """Looks up an order's amount, delivery status, and age in days from the database.

    Args:
        order_id: The unique order identifier (e.g., 'ORD-101').
    """
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        log_validation_failure(
            "lookup_order",
            {"order_id": order_id},
            "Order ID not found in database",
            "Model queried a non-existent order ID",
        )
        return {"status": "ERROR", "message": f"Order '{order_id}' does not exist."}
    return {"status": "FOUND", "order_id": order_id.upper(), **order}


def process_refund(order_id: str, refund_amount: float, reason: str) -> dict:
    """Processes a customer refund after validating schema and business rules.

    Args:
        order_id: The unique order identifier (e.g., 'ORD-101').
        refund_amount: Amount in INR to refund to the customer.
        reason: Reason for the refund request.
    """
    raw_args = {"order_id": order_id, "refund_amount": refund_amount, "reason": reason}

    # LAYER 1: Schema & Type Validation
    try:
        validated = RefundRequest(**raw_args)
    except ValidationError as e:
        log_validation_failure(
            "process_refund",
            raw_args,
            f"Schema Error: {e}",
            "Invalid argument types/bounds",
        )
        return {"status": "REJECTED", "reason": f"Invalid input format: {e}"}

    # LAYER 2: Business Rule Validation Against Source Data
    order = ORDERS_DB.get(validated.order_id.upper())
    if not order:
        log_validation_failure(
            "process_refund",
            raw_args,
            "Unknown Order ID",
            "Attempted refund on missing order",
        )
        return {"status": "REJECTED", "reason": f"Order {validated.order_id} does not exist."}

    if order["status"] != "DELIVERED":
        rule = f"Order status is '{order['status']}' (must be DELIVERED before refund)"
        log_validation_failure(
            "process_refund",
            raw_args,
            rule,
            "Premature refund on undelivered order",
        )
        return {"status": "REJECTED", "reason": rule}

    if validated.refund_amount > order["amount"]:
        rule = f"Requested ₹{validated.refund_amount} exceeds original order total of ₹{order['amount']}"
        log_validation_failure(
            "process_refund",
            raw_args,
            rule,
            "Over-refund attempt",
        )
        return {"status": "REJECTED", "reason": rule}

    if order["days_ago"] > 30:
        rule = f"Order is {order['days_ago']} days old (exceeds 30-day return window)"
        log_validation_failure(
            "process_refund",
            raw_args,
            rule,
            "Expired return window",
        )
        return {"status": "REJECTED", "reason": rule}

    return {
        "status": "APPROVED",
        "order_id": validated.order_id.upper(),
        "refunded_inr": validated.refund_amount,
        "message": "Refund processed successfully.",
    }



test_prompts = [
    # Scenario A: Valid request (passes both schema and business rules)
    "Please refund ₹4500 for order ORD-101 because the spacebar key is sticking.",
    # Scenario B: Valid types, but breaks business rule (amount > order total)
    "Refund ₹9000 for order ORD-101 immediately.",
    # Scenario C: Valid types, but breaks 30-day policy rule
    "I want a ₹1200 refund for order ORD-102, it stopped working.",
    # Scenario D: Valid types, but breaks delivery status rule (IN_TRANSIT)
    "Refund ₹1800 for order ORD-103, I changed my mind before delivery.",
]

for i, prompt in enumerate(test_prompts, start=1):
    print(f"\n{'='*60}\n🧪 TEST {i}: {prompt}\n{'='*60}")
    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a customer support billing agent. Always check order details with "
                "lookup_order before calling process_refund. Never approve a refund yourself—rely "
                "strictly on the tool response."
            ),
            tools=[lookup_order, process_refund],
            temperature=0.0,
        ),
    )
    response = chat.send_message(prompt)
    print(f"🤖 Agent Reply:\n{response.text}")