# chat_UI/mongo_utils.py

import os
import re
import uuid
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from bson.errors import InvalidId
from datetime import datetime, time

# --- Cấu hình tên Database và Collection ---
MONGO_DATABASE_NAME = os.getenv("MONGO_DB", "kieu_bot")
# Chat history must never be stored with RAG chunks. A dedicated collection
# avoids polluting retrieval results and lets retention policies be applied.
CHAT_COLLECTION_NAME = os.getenv("MONGO_CHAT_COLLECTION", "chat_messages")
CONVERSATION_COLLECTION_NAME = os.getenv("MONGO_CONVERSATION_COLLECTION", "chat_conversations")
INTERACTION_COLLECTION_NAME = os.getenv("MONGO_INTERACTION_COLLECTION", "chat_interactions")
LEGACY_CONVERSATION_ID = "legacy"
_UNSET = object()

_mongo_client = None


def get_mongo_client():
    """Tạo hoặc tái sử dụng một kết nối MongoClient."""
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("Biến môi trường MONGO_URI chưa được thiết lập.")
        timeout_ms = int(os.getenv("MONGO_TIMEOUT_MS", "2500"))
        _mongo_client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=max(timeout_ms, 3000),
            maxPoolSize=20,
        )
    return _mongo_client


def get_chat_collection():
    """Lấy collection chat từ MongoDB."""
    client = get_mongo_client()
    db = client[MONGO_DATABASE_NAME]
    return db[CHAT_COLLECTION_NAME]


def get_conversation_collection():
    client = get_mongo_client()
    return client[MONGO_DATABASE_NAME][CONVERSATION_COLLECTION_NAME]


def get_interaction_collection():
    client = get_mongo_client()
    return client[MONGO_DATABASE_NAME][INTERACTION_COLLECTION_NAME]


def normalize_conversation_id(value):
    candidate = str(value or LEGACY_CONVERSATION_ID).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", candidate):
        raise ValueError("invalid conversation id")
    return candidate


def _conversation_message_filter(user, conversation_id):
    conversation_id = normalize_conversation_id(conversation_id)
    query = {"user_id": user.id}
    if conversation_id == LEGACY_CONVERSATION_ID:
        query["$or"] = [
            {"conversation_id": LEGACY_CONVERSATION_ID},
            {"conversation_id": {"$exists": False}},
        ]
    else:
        query["conversation_id"] = conversation_id
    return query


def create_conversation(user, title="Cuộc trò chuyện mới"):
    conversation_id = uuid.uuid4().hex
    timestamp = datetime.now()
    get_conversation_collection().insert_one(
        {
            "conversation_id": conversation_id,
            "user_id": user.id,
            "username": user.username,
            "title": str(title or "Cuộc trò chuyện mới").strip()[:80],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
    )
    return conversation_id


def ensure_conversation(user, conversation_id, title=None):
    conversation_id = normalize_conversation_id(conversation_id)
    timestamp = datetime.now()
    get_conversation_collection().update_one(
        {"user_id": user.id, "conversation_id": conversation_id},
        {
            "$set": {"updated_at": timestamp},
            "$setOnInsert": {
                "username": user.username,
                "title": str(title or "Cuộc trò chuyện mới").strip()[:80],
                "created_at": timestamp,
            },
        },
        upsert=True,
    )
    return conversation_id


def list_conversations(user):
    collection = get_conversation_collection()
    conversations = list(collection.find({"user_id": user.id}).sort("updated_at", DESCENDING))
    payload = [
        {
            "id": item["conversation_id"],
            "title": item.get("title") or "Mạch đọc",
            "created_at": item.get("created_at").isoformat() if item.get("created_at") else None,
            "updated_at": item.get("updated_at").isoformat() if item.get("updated_at") else None,
        }
        for item in conversations
    ]
    legacy_filter = _conversation_message_filter(user, LEGACY_CONVERSATION_ID)
    if get_chat_collection().count_documents(legacy_filter) and not any(item["id"] == LEGACY_CONVERSATION_ID for item in payload):
        first = get_chat_collection().find_one(legacy_filter, sort=[("created_at", 1)]) or {}
        text = str(first.get("content") or "Mạch đọc trước đây").replace("\n", " ").strip()
        payload.append(
            {
                "id": LEGACY_CONVERSATION_ID,
                "title": (text[:47] + "…") if len(text) > 48 else text,
                "created_at": first.get("created_at").isoformat() if first.get("created_at") else None,
                "updated_at": None,
            }
        )
    return payload


def rename_conversation(user, conversation_id, title):
    conversation_id = normalize_conversation_id(conversation_id)
    title = str(title or "").strip()[:80]
    if not title:
        raise ValueError("title is required")
    result = get_conversation_collection().update_one(
        {"user_id": user.id, "conversation_id": conversation_id},
        {"$set": {"title": title, "updated_at": datetime.now()}},
        upsert=conversation_id == LEGACY_CONVERSATION_ID,
    )
    return result.matched_count > 0 or result.upserted_id is not None


def delete_conversation(user, conversation_id):
    conversation_id = normalize_conversation_id(conversation_id)
    deleted_messages = get_chat_collection().delete_many(_conversation_message_filter(user, conversation_id)).deleted_count
    get_conversation_collection().delete_one({"user_id": user.id, "conversation_id": conversation_id})
    get_interaction_collection().delete_many({"user_id": user.id, "conversation_id": conversation_id})
    return deleted_messages


def save_message_to_mongo(user, role, content, meta=None, conversation_id=None):
    """Lưu một tin nhắn vào MongoDB."""
    collection = get_chat_collection()
    conversation_id = normalize_conversation_id(conversation_id)
    message_doc = {
        "user_id": user.id,  # Hoặc user.username, tùy bạn muốn định danh thế nào
        "username": user.username,
        "role": role,
        "content": content,
        "meta": meta or {},
        "conversation_id": conversation_id,
        "created_at": datetime.now(),
    }
    result = collection.insert_one(message_doc)
    ensure_conversation(user, conversation_id)
    return str(result.inserted_id)


def get_history_for_api(user, conversation_id=None):
    """Lấy lịch sử chat đầy đủ để trả về cho API frontend."""
    collection = get_chat_collection()
    # Lấy tất cả tin nhắn của user, sắp xếp theo thời gian
    conversation_id = normalize_conversation_id(conversation_id)
    messages = collection.find(_conversation_message_filter(user, conversation_id)).sort("created_at", 1)

    interactions = {
        item["message_id"]: item
        for item in get_interaction_collection().find(
            {"user_id": user.id, "conversation_id": conversation_id}
        )
    }

    # Chuyển đổi định dạng để tương thích với JSON
    return [
        {
            "id": str(m["_id"]),
            "role": m["role"],
            "content": m["content"],
            "meta": m.get("meta", {}),
            "ts": m["created_at"].isoformat(),
            "actions": {
                "saved": bool(interactions.get(str(m["_id"]), {}).get("saved")),
                "feedback": interactions.get(str(m["_id"]), {}).get("feedback"),
            },
        }
        for m in messages
    ]


def get_history_for_bot(user, limit=12, conversation_id=None):
    """Lấy lịch sử chat đã được định dạng để gửi cho bot."""
    collection = get_chat_collection()
    # Lấy `limit` tin nhắn cuối cùng
    messages = collection.find(_conversation_message_filter(user, conversation_id)).sort("created_at", DESCENDING).limit(limit)

    # Bot cần định dạng (role, content) và theo thứ tự từ cũ -> mới
    history_tuples = [(m["role"], m["content"]) for m in messages]
    return list(reversed(history_tuples))


def clear_user_history(user, conversation_id=None):
    """Xóa toàn bộ lịch sử chat của một người dùng."""
    collection = get_chat_collection()
    if conversation_id is None:
        collection.delete_many({"user_id": user.id})
        get_conversation_collection().delete_many({"user_id": user.id})
        get_interaction_collection().delete_many({"user_id": user.id})
        return
    delete_conversation(user, conversation_id)


def update_message_actions(user, message_id, *, saved=_UNSET, feedback=_UNSET):
    try:
        object_id = ObjectId(str(message_id))
    except (InvalidId, TypeError):
        raise ValueError("invalid message id")
    message = get_chat_collection().find_one({"_id": object_id, "user_id": user.id, "role": "assistant"})
    if not message:
        return None
    conversation_id = normalize_conversation_id(message.get("conversation_id"))
    update = {"updated_at": datetime.now(), "conversation_id": conversation_id}
    if saved is not _UNSET:
        update["saved"] = bool(saved)
    if feedback is not _UNSET:
        if feedback not in {"up", "down", None}:
            raise ValueError("invalid feedback")
        update["feedback"] = feedback
    get_interaction_collection().update_one(
        {"user_id": user.id, "message_id": str(object_id)},
        {"$set": update, "$setOnInsert": {"created_at": datetime.now()}},
        upsert=True,
    )
    state = get_interaction_collection().find_one({"user_id": user.id, "message_id": str(object_id)}) or {}
    return {"saved": bool(state.get("saved")), "feedback": state.get("feedback")}


def count_user_messages_today(user):
    """Đếm số tin nhắn của người dùng trong ngày hôm nay."""
    collection = get_chat_collection()
    today_start = datetime.combine(datetime.today(), time.min)
    today_end = datetime.combine(datetime.today(), time.max)

    count = collection.count_documents(
        {
            "user_id": user.id,
            "role": "user",  # Chỉ đếm tin nhắn của người dùng
            "created_at": {"$gte": today_start, "$lte": today_end},
        }
    )
    return count
