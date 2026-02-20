# MongoDB (Motor async). Query-only: list_databases, list_collections, query_documents.
# LLM uses these to read from Atlas (e.g. CrewgleAI_Store, Sports Items). No conversation storage.
# Config from AI_Bot/.env: MONGODB_URI.

import os
import asyncio
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient

# Load .env from AI_Bot/
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

_SKIP_DBS = frozenset({"admin", "local", "config"})

_SEARCH_STOPWORDS = frozenset({
    "do", "we", "have", "any", "in", "our", "the", "a", "an", "is", "are", "what",
    "which", "does", "can", "you", "your", "my", "me", "i", "it", "this", "that",
    "there", "here", "of", "to", "for", "with", "on", "at", "data", "store", "stock",
})


def _get_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(MONGODB_URI)


async def ping(client: AsyncIOMotorClient) -> bool:
    """Check MongoDB connectivity."""
    try:
        await client.admin.command("ping")
        return True
    except Exception as e:
        import sys
        print(f"MongoDB connection error: {e}", file=sys.stderr)
        return False


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


def _word_matches_text(word: str, text_lower: str) -> bool:
    w = word.lower().strip()
    if not w:
        return True
    if w in text_lower:
        return True
    if w.endswith("s") and len(w) > 1 and w[:-1] in text_lower:
        return True
    if w + "s" in text_lower:
        return True
    return False


def _doc_matches_search(formatted: str, words: list[str]) -> bool:
    if not formatted:
        return False
    filtered = [w for w in words if w and w.lower() not in _SEARCH_STOPWORDS]
    if not filtered:
        return False
    lower = formatted.lower()
    return all(_word_matches_text(w, lower) for w in filtered)


async def list_databases(client: AsyncIOMotorClient | None = None) -> list[str]:
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


async def query_documents(
    client: AsyncIOMotorClient | None,
    database: str,
    collection: str,
    search_term: str,
) -> str:
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


# --- Sync wrappers for LLM tools ---

def ping_sync() -> bool:
    async def _():
        client = _get_client()
        try:
            return await ping(client)
        finally:
            client.close()
    return asyncio.run(_())


def list_databases_sync() -> list[str]:
    try:
        return asyncio.run(list_databases(None))
    except Exception as e:
        import sys
        print(f"  → list_databases error: {e}\n", file=sys.stderr)
        return []


def list_collections_sync(database: str) -> list[str]:
    try:
        return asyncio.run(list_collections(None, database))
    except Exception as e:
        import sys
        print(f"  → list_collections error: {e}\n", file=sys.stderr)
        return []


def query_documents_sync(database: str, collection: str, search_term: str) -> str:
    try:
        return asyncio.run(query_documents(None, database, collection, search_term))
    except Exception as e:
        import sys
        print(f"  → query_documents error: {e}\n", file=sys.stderr)
        return f"Database error: {e}"
