"""OpenAI LLM for voice bot. Uses function calling to query MongoDB (any DB/collection) based on user questions."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import OpenAI

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = "You are TONY, a helpful voice assistant. Reply in one or two short sentences."

# LLM can query MongoDB Atlas when user asks about stored data; otherwise answer from training.
SYSTEM_PROMPT_WITH_TOOLS = (
    "You are TONY, a helpful voice assistant. You have access to MongoDB Atlas with tools to read the user's data "
    "(e.g. CrewgleAI_Store, Sports Items, products, inventory). Use these tools ONLY for questions about stored data: "
    "products, sports items, prices, stock, warranty, origin, or similar. For those questions, call the tools and base your answer on the data returned. "
    "For all other topics (people, places, general knowledge, chit-chat), answer from your training and DO NOT call the tools. "
    "Reply in one or two short sentences."
)

# OpenAI function definitions (JSON Schema)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_databases",
            "description": "List all database names on the MongoDB server (cloud or local). Use this to see what data is available.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_collections",
            "description": "List all collection names in a database. Call list_databases first to get database names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "The database name (e.g. 'CrewgleAI_Store').",
                    },
                },
                "required": ["database"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_documents",
            "description": "Query documents in MongoDB. Search for documents that contain all words in search_term (case-insensitive). To search everywhere, use empty strings for database and collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name (e.g. 'CrewgleAI_Store'), or '' to search all databases.",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection name (e.g. 'Sports Items', 'Product'), or '' to search all collections.",
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Key product/category terms only, e.g. 'laptop', 'Logitech MX Master', 'mouse price'. Use 1-3 keywords, not the full question.",
                    },
                },
                "required": ["database", "collection", "search_term"],
            },
        },
    },
]


def list_databases() -> list[str]:
    """List all database names on the MongoDB Atlas server. Use this to see what data is available."""
    from db import list_databases_sync
    return list_databases_sync()


def list_collections(database: str) -> list[str]:
    """List all collection names in a database. Call list_databases first to get database names."""
    from db import list_collections_sync
    return list_collections_sync(database)


def query_documents(database: str, collection: str, search_term: str) -> str:
    """Query documents in MongoDB. Search for documents that contain all words in search_term (case-insensitive)."""
    from db import query_documents_sync
    return query_documents_sync(database, collection, search_term)


def _get_client() -> OpenAI:
    """Return OpenAI client. Uses OPENAI_API_KEY from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI API")
    return OpenAI(api_key=api_key)


def ask_llm(
    prompt: str,
    system: str | None = None,
    use_tools: bool = False,
) -> str:
    """Send user text to OpenAI and return the assistant reply.

    If use_tools is True, tools are available but only used when the prompt looks like
    a product/inventory question; otherwise the model answers from its training data.
    """

    def looks_like_data_question(text: str) -> bool:
        t = (text or "").lower()
        keywords = (
            "price", "cost", "how much", "rs", "₹", "$", "dollar", "origin", "warrenty", "warranties",
            "monitor", "mouse", "keyboard", "laptop", "headphone", "earphone", "ssd", "hdd", "gpu", "graphics card",
            "jeans", "pants", "shirt", "shoes", "shoe", "heels", "sneakers",
            "sport", "sports", "ball", "cricket", "football", "item", "items", "colour", "brand", "size",
            "do we have", "in stock", "inventory", "product", "products", "database", "data", "store",
            "crewgleai_store", "crewgle", "sports items", "fashion items",
        )
        return any(k in t for k in keywords)

    use_db_tools = bool(use_tools and looks_like_data_question(prompt))

    client = _get_client()
    messages = [{"role": "user", "content": prompt}]
    system_content = SYSTEM_PROMPT_WITH_TOOLS if use_db_tools else (system or SYSTEM_PROMPT)
    messages.insert(0, {"role": "system", "content": system_content})

    if not use_db_tools:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=150,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    # With tools: loop until we get a final text reply
    while True:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            max_tokens=150,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            return content.strip()

        # Append assistant message with tool calls
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
        })

        # Execute tools and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
            except json.JSONDecodeError:
                args = {}

            if name == "list_databases":
                result = list_databases()
            elif name == "list_collections":
                result = list_collections(args.get("database", ""))
            elif name == "query_documents":
                result = query_documents(
                    args.get("database", ""),
                    args.get("collection", ""),
                    args.get("search_term", ""),
                )
            else:
                result = "Unknown tool."

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result) if not isinstance(result, list) else json.dumps(result),
            })
