# V2 — MongoDB (Motor async). One conversation = one document with a "turns" array.
# Use create_session() / insert_turn() from async code, or *_sync() from sync code.

import os
import asyncio
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

# Config: env MONGODB_URI (default local), VOICE_BOT_DB (default voice_bot)
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("VOICE_BOT_DB", "voice_bot")

# Single collection: one document per conversation (all turns inside)
COLL_CONVERSATIONS = "conversations"

# System DBs to skip when listing or searching "all" documents.
_SKIP_DBS = frozenset({"admin", "local", "config"})

# Question/stop words to strip from search_term so we match on product terms only
_SEARCH_STOPWORDS = frozenset({
    "do", "we", "have", "any", "in", "our", "the", "a", "an", "is", "are", "what",
    "which", "does", "can", "you", "your", "my", "me", "i", "it", "this", "that",
    "there", "here", "of", "to", "for", "with", "on", "at", "data", "store", "stock",
})


def _get_client() -> AsyncIOMotorClient:
    """Return a new Motor client (use in async context or per sync call)."""
    return AsyncIOMotorClient(MONGODB_URI)


async def ping(client: AsyncIOMotorClient) -> bool:
    """Check MongoDB connectivity."""
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def create_session(client: AsyncIOMotorClient | None = None) -> str:
    """Create one conversation document (turns=[]) and return its _id as string."""
    own_client = client is None
    if own_client:
        client = _get_client()
    try:
        db = client[DB_NAME]
        coll = db[COLL_CONVERSATIONS]
        now = datetime.now(timezone.utc)
        doc = {"created_at": now, "updated_at": now, "turns": [], "metadata": {}}
        result = await coll.insert_one(doc)
        return str(result.inserted_id)
    finally:
        if own_client:
            client.close()


async def insert_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    *,
    client: AsyncIOMotorClient | None = None,
    model: str | None = None,
) -> None:
    """Append one turn to the conversation document ( $push to turns )."""
    own_client = client is None
    if own_client:
        client = _get_client()
    try:
        db = client[DB_NAME]
        coll = db[COLL_CONVERSATIONS]
        now = datetime.now(timezone.utc)
        turn = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "created_at": now,
        }
        if model:
            turn["model"] = model
        await coll.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"turns": turn}, "$set": {"updated_at": now}},
        )
    finally:
        if own_client:
            client.close()


def _format_knowledge_doc(doc: dict) -> str:
    """Turn one document into a readable string (skip _id)."""
    parts = []
    for k, v in doc.items():
        if k == "_id":
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(f"{k}: {s}")
    return "\n".join(parts) if parts else ""


async def list_databases(client: AsyncIOMotorClient | None = None) -> list[str]:
    """List all database names on the MongoDB server (excluding admin, local, config)."""
    own_client = client is None
    if own_client:
        client = _get_client()
    try:
        names = await client.list_database_names()
        return [n for n in names if n not in _SKIP_DBS]
    finally:
        if own_client:
            client.close()


async def list_collections(client: AsyncIOMotorClient | None, database: str) -> list[str]:
    """List all collection names in the given database."""
    own_client = client is None
    if own_client:
        client = _get_client()
    try:
        if not (database and database.strip()):
            return []
        db = client[database.strip()]
        names = await db.list_collection_names()
        return list(names)
    finally:
        if own_client:
            client.close()


def _word_matches_text(word: str, text_lower: str) -> bool:
    """True if word (or its plural/singular form) appears in text. Handles laptop/laptops etc."""
    w = word.lower().strip()
    if not w:
        return True
    if w in text_lower:
        return True
    # Plural/singular: "laptop" matches "laptops", "laptops" matches "laptop"
    if w.endswith("s") and len(w) > 1 and w[:-1] in text_lower:
        return True
    if w + "s" in text_lower:
        return True
    return False


def _doc_matches_search(formatted: str, words: list[str]) -> bool:
    """True if all (non-stopword) words appear in formatted (case-insensitive). Handles plurals."""
    if not formatted:
        return False
    # Filter stopwords so "Do we have any laptop" -> ["laptop"]
    filtered = [w for w in words if w and w.lower() not in _SEARCH_STOPWORDS]
    if not filtered:
        return False
    lower = formatted.lower()
    return all(_word_matches_text(w, lower) for w in filtered)


async def query_documents(
    client: AsyncIOMotorClient | None,
    database: str,
    collection: str,
    search_term: str,
) -> str:
    """Query documents in database.collection. If database or collection is empty, search ALL databases and
    collections on the server (excluding system DBs) and return results labeled by source.
    Documents are matched when all words in search_term appear in the document text (case-insensitive)."""
    own_client = client is None
    if own_client:
        client = _get_client()
    try:
        words = [w.strip() for w in (search_term or "").strip().split() if w.strip()]
        if not words:
            return "No search term provided."

        async def search_in_coll(db_name: str, coll_name: str) -> list[str]:
            out = []
            try:
                cursor = client[db_name][coll_name].find({})
                async for doc in cursor:
                    formatted = _format_knowledge_doc(doc)
                    if formatted and _doc_matches_search(formatted, words):
                        out.append(formatted)
            except Exception:
                pass
            return out

        if (database or "").strip() and (collection or "").strip():
            db_name = database.strip()
            coll_name = collection.strip()
            chunks = await search_in_coll(db_name, coll_name)
            if not chunks:
                return f"No matching documents in {db_name}.{coll_name}."
            return "\n\n---\n\n".join(chunks)
        else:
            db_list = await list_databases(client)
            all_results = []
            for db_name in db_list:
                coll_list = await list_collections(client, db_name)
                for coll_name in coll_list:
                    chunks = await search_in_coll(db_name, coll_name)
                    for c in chunks:
                        all_results.append(f"[{db_name}.{coll_name}]\n{c}")
            if not all_results:
                return "No matching documents found in any database or collection."
            return "\n\n---\n\n".join(all_results)
    finally:
        if own_client:
            client.close()


# --- Sync wrappers for use from live_transcript (sync main loop) ---

def create_session_sync() -> str:
    """Create a session from sync code. Returns session_id string."""
    return asyncio.run(create_session(None))


def insert_turn_sync(
    session_id: str,
    user_text: str,
    assistant_text: str,
    model: str | None = None,
) -> None:
    """Insert one turn from sync code. Logs and swallows DB errors so pipeline keeps running."""
    try:
        asyncio.run(insert_turn(session_id, user_text, assistant_text, model=model))
    except Exception as e:
        import sys
        print(f"  → DB write error (turn not saved): {e}\n", file=sys.stderr)


def ping_sync() -> bool:
    """Check MongoDB from sync code."""
    async def _():
        client = _get_client()
        try:
            return await ping(client)
        finally:
            client.close()
    return asyncio.run(_())


def list_databases_sync() -> list[str]:
    """List all database names (excluding system DBs). On error returns []."""
    try:
        return asyncio.run(list_databases(None))
    except Exception as e:
        import sys
        print(f"  → list_databases error: {e}\n", file=sys.stderr)
        return []


def list_collections_sync(database: str) -> list[str]:
    """List all collection names in the given database. On error returns []."""
    try:
        return asyncio.run(list_collections(None, database))
    except Exception as e:
        import sys
        print(f"  → list_collections error: {e}\n", file=sys.stderr)
        return []


def query_documents_sync(database: str, collection: str, search_term: str) -> str:
    """Query documents in database.collection, or search all DBs/collections if database or collection is empty."""
    try:
        return asyncio.run(query_documents(None, database, collection, search_term))
    except Exception as e:
        import sys
        print(f"  → query_documents error: {e}\n", file=sys.stderr)
        return f"Database error: {e}"
