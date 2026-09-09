# chat_UI/mongo_utils.py

import os
from pymongo import MongoClient, DESCENDING
from datetime import datetime, time

# --- Cấu hình tên Database và Collection ---
MONGO_DATABASE_NAME = os.getenv("MONGO_DB", "kieu_bot")
# Chat history must never be stored with RAG chunks. A dedicated collection
# avoids polluting retrieval results and lets retention policies be applied.
CHAT_COLLECTION_NAME = os.getenv("MONGO_CHAT_COLLECTION", "chat_messages")

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


def save_message_to_mongo(user, role, content, meta=None):
    """Lưu một tin nhắn vào MongoDB."""
    collection = get_chat_collection()
    message_doc = {
        "user_id": user.id,  # Hoặc user.username, tùy bạn muốn định danh thế nào
        "username": user.username,
        "role": role,
        "content": content,
        "meta": meta or {},
        "created_at": datetime.now(),
    }
    collection.insert_one(message_doc)


def get_history_for_api(user):
    """Lấy lịch sử chat đầy đủ để trả về cho API frontend."""
    collection = get_chat_collection()
    # Lấy tất cả tin nhắn của user, sắp xếp theo thời gian
    messages = collection.find({"user_id": user.id}).sort("created_at", 1)

    # Chuyển đổi định dạng để tương thích với JSON
    return [
        {"role": m["role"], "content": m["content"], "meta": m.get("meta", {}), "ts": m["created_at"].isoformat()}
        for m in messages
    ]


def get_history_for_bot(user, limit=12):
    """Lấy lịch sử chat đã được định dạng để gửi cho bot."""
    collection = get_chat_collection()
    # Lấy `limit` tin nhắn cuối cùng
    messages = collection.find({"user_id": user.id}).sort("created_at", DESCENDING).limit(limit)

    # Bot cần định dạng (role, content) và theo thứ tự từ cũ -> mới
    history_tuples = [(m["role"], m["content"]) for m in messages]
    return list(reversed(history_tuples))


def clear_user_history(user):
    """Xóa toàn bộ lịch sử chat của một người dùng."""
    collection = get_chat_collection()
    collection.delete_many({"user_id": user.id})


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
