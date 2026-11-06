
"""
Banking Seminar · Day 1 (LangChain) — Step 4: Tools and a simple agent.

A TOOL is just a Python function the model is allowed to call.
An AGENT is a model that decides WHICH tool to call and WHEN, reads the
result, and then writes the final answer.

Here the model can check a balance, deposit, withdraw, or transfer money
between accounts.
(On Day 2 we add a HUMAN approval step before money leaves the account.)

Run it:   python banking/day_1/04_tools_fixed.py   (type 'quit' to exit)
"""

import json
import os
from pathlib import Path
import pprint

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parent / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# The account details live in a plain JSON file -- our "database without a
# database". Open banking/data/account.json and watch the balance change as
# you deposit and withdraw. (On Day 2 we also show a REAL vector database.)
ACCOUNT_FILE = Path(__file__).resolve().parent / "data" / "account.json"


def load_data() -> dict:
    """Read the full {"accounts": [...]} structure from the JSON file."""
    return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))


def save_data(data: dict) -> None:
    """Write the full {"accounts": [...]} structure back to the JSON file."""
    ACCOUNT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_account(data: dict, identifier: str) -> dict:
    """Look up one account by name or account number (case-insensitive)."""
    identifier = identifier.strip().lower()
    for account in data["accounts"]:
        if (
            account["name"].lower() == identifier
            or account["account_number"].lower() == identifier
        ):
            return account
    raise ValueError(f"No account found matching '{identifier}'.")


def check_balance(account: str) -> str:
    """Return the current balance for the named account or account number."""
    data = load_data()
    acc = find_account(data, account)
    return f"{acc['name']}'s balance is UGX {acc['balance']:,.0f}."


def deposit(amount: float, account: str) -> str:
    """Add money to the named account. 'amount' must be positive."""
    if amount <= 0:
        return "Deposit amount must be positive."
    data = load_data()
    acc = find_account(data, account)
    acc["balance"] += amount
    save_data(data)
    return f"Deposited UGX {amount:,.0f} into {acc['name']}'s account. New balance: UGX {acc['balance']:,.0f}."


def withdraw(amount: float, account: str) -> str:
    """Take money out of the named account."""
    if amount <= 0:
        return "Withdrawal amount must be positive."
    data = load_data()
    acc = find_account(data, account)
    if amount > acc["balance"]:
        return f"Insufficient funds. {acc['name']}'s balance is only UGX {acc['balance']:,.0f}."
    acc["balance"] -= amount
    save_data(data)
    return f"Withdrew UGX {amount:,.0f} from {acc['name']}'s account. New balance: UGX {acc['balance']:,.0f}."


def account_name(account: str) -> str:
    """Confirm the account name for the given name or account number."""
    data = load_data()
    acc = find_account(data, account)
    return f"Your account name is {acc['name']}."


def transfer_amount(amount: float, acc_from: str, acc_to: str) -> str:
    """Transfer money from one account to another."""
    if amount <= 0:
        return "Transfer amount has to be positive."

    data = load_data()
    from_account = find_account(data, acc_from)
    to_account = find_account(data, acc_to)

    if from_account is to_account:
        return "Cannot transfer to the same account."
    if amount > from_account["balance"]:
        return f"Insufficient funds. {from_account['name']}'s balance is only UGX {from_account['balance']:,.0f}."

    from_account["balance"] -= amount
    to_account["balance"] += amount
    save_data(data)

    return (
        f"UGX {amount:,.0f} has been withdrawn from {from_account['name']} "
        f"and deposited to {to_account['name']}."
    )


agent = create_agent(
    model=model,
    tools=[check_balance, deposit, withdraw, account_name, transfer_amount],
    system_prompt=(
        "You are a helpful bank assistant. Use a tool to check the balance, "
        "deposit, withdraw, or transfer money -- never make up numbers. "
        "Ask the user which account (by name or account number) if it isn't "
        "clear. Confirm what you did in one short sentence."
    ),
)


def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("Assistant:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    print("Bank Assistant with tools (LangChain). Type 'quit' to exit.")
    print("Examples:  What's my balance?  |  Deposit 100000  |  Withdraw 50000\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        ask(q)
