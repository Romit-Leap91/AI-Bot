"""Ollama LLM for voice bot. Uses tool calling to query MongoDB (any DB/collection) based on user questions."""
import ollama

OLLAMA_MODEL = "qwen3:latest"  # qwen3 has good tool-calling support
SYSTEM_PROMPT = "You are TONY, a helpful voice assistant. Reply in one or two short sentences."

# Document-oriented: LLM decides which database/collection to query based on the user's question.
SYSTEM_PROMPT_WITH_TOOLS = (
    "You are TONY, a helpful voice assistant. You have access to MongoDB on localhost (all databases and collections). "
    "Use these tools to answer questions about data the user might have in their documents:\n"
    "1. list_databases() - list all databases (excluding system DBs).\n"
    "2. list_collections(database) - list all collections in that database.\n"
    "3. query_documents(database, collection, search_term) - search documents. If you know the right database and collection, use them. "
    "If you are unsure, use empty strings for database and collection to search ALL databases and collections. "
    "Use the search_term from the user's question (e.g. product name, topic, keyword).\n"
    "Answer using ONLY the data returned by the tools. If no matching documents are found, say so. Do not guess or invent data. "
    "For general chat (greetings, thanks), reply normally without calling tools. Reply in one or two short sentences."
)


def list_databases() -> list[str]:
    """List all database names on the MongoDB server (localhost). Use this to see what data is available."""
    from db import list_databases_sync
    return list_databases_sync()


def list_collections(database: str) -> list[str]:
    """List all collection names in a database. Call list_databases first to get database names.

    Args:
        database: The database name (e.g. 'Crewgle_Store', 'voice_bot').
    """
    from db import list_collections_sync
    return list_collections_sync(database)


def query_documents(database: str, collection: str, search_term: str) -> str:
    """Query documents in MongoDB. Search for documents that contain all words in search_term (case-insensitive).
    To search everywhere, use empty strings for database and collection.

    Args:
        database: Database name (e.g. 'Crewgle_Store'), or '' to search all databases.
        collection: Collection name (e.g. 'Product'), or '' to search all collections.
        search_term: Words to search for in documents (e.g. 'Logitech MX Master price', 'refund policy').
    """
    from db import query_documents_sync
    return query_documents_sync(database, collection, search_term)


def ask_llm(
    prompt: str,
    system: str | None = None,
    context: str | None = None,
    use_tools: bool = False,
) -> str:
    """Send user text to Ollama and return the assistant reply.
    If use_tools is True, the model can call query_product_database; we run the tool and feed the result back (agent loop)."""
    messages = [{"role": "user", "content": prompt}]
    system_content = SYSTEM_PROMPT_WITH_TOOLS if use_tools else (system or SYSTEM_PROMPT)
    if not use_tools and context is not None and context.strip():
        system_content += (
            "\n\nCRITICAL - Database rules:\n"
            "The following is the ONLY source of truth for product names, prices, and factual data.\n"
            "You MUST answer such questions using ONLY the data below. Do NOT use your training data or guess.\n"
            "If the answer is not in the data below, say: \"I don't have that in my database.\"\n"
            "Reply in one or two short sentences.\n\n"
            "Data from database:\n" + context.strip()
        )
    elif not use_tools and context is not None and "[Knowledge base is empty" in context:
        system_content += (
            "\n\nYou have no product/price data loaded. "
            "For any product or price question, say: \"I don't have that in my database.\""
        )
    messages.insert(0, {"role": "system", "content": system_content})

    if not use_tools:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return _content_from_response(response)

    # Tool-calling agent loop: model may call list_databases, list_collections, query_documents.
    # If the model doesn't call any tool but the user might be asking about data, we proactively search all and re-ask.
    tools = [list_databases, list_collections, query_documents]
    first_turn = True
    TOOL_FUNCS = {"list_databases": list_databases, "list_collections": list_collections, "query_documents": query_documents}

    def _might_need_data(text: str) -> bool:
        """Heuristic: user might be asking about something in their documents."""
        t = (text or "").lower()
        return any(k in t for k in (
            "price", "cost", "how much", "product", "refund", "policy", "database", "document",
            "what is", "tell me about", "do you have", "in your", "in the db", "in the database",
        )) or "?" in (text or "")

    while True:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
        )
        msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if msg is None:
            return ""
        if hasattr(msg, "tool_calls"):
            tool_calls = list(msg.tool_calls or [])
            content = getattr(msg, "content", None) or ""
        else:
            tool_calls = list(msg.get("tool_calls") or [])
            content = (msg.get("content") or "").strip()

        def _tc_name(tc):
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            return fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")

        def _tc_args(tc):
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", None) or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return args if isinstance(args, dict) else {}

        assistant_msg = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": _tc_name(tc),
                        "arguments": _tc_args(tc),
                    }
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)
        else:
            # Proactive fallback: search all DBs/collections when user might be asking about data
            if first_turn and _might_need_data(prompt):
                db_result = query_documents("", "", prompt)
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "query_documents",
                            "arguments": {"database": "", "collection": "", "search_term": prompt},
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_name": "query_documents",
                    "content": db_result,
                })
                first_turn = False
                continue
            messages.append(assistant_msg)
            return (content or "").strip()

        first_turn = False
        for tc in tool_calls:
            name = _tc_name(tc)
            args = _tc_args(tc)
            fn = TOOL_FUNCS.get(name)
            if fn is None:
                result = "Unknown tool."
            elif name == "list_databases":
                result = str(fn())
            elif name == "list_collections":
                result = str(fn(args.get("database", "")))
            elif name == "query_documents":
                result = fn(
                    args.get("database", ""),
                    args.get("collection", ""),
                    args.get("search_term", ""),
                )
            else:
                result = "Unknown tool."
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})


def _content_from_response(response) -> str:
    """Extract content string from Ollama chat response (dict or object)."""
    msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return (msg.get("content") or "").strip()
    return (getattr(msg, "content", None) or "").strip()
