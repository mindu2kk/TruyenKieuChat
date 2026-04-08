# chat_UI/views.py
import json
from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.timezone import now
from django.conf import settings
from datetime import date

from .models import UserProfile

from app.orchestrator import answer_with_router
from app.generation import is_gemini_configured
from app.poem_tools import poem_ready
import logging, traceback
logger = logging.getLogger(__name__)

from .mongo_utils import (
    save_message_to_mongo,
    get_history_for_api,
    get_history_for_bot,
    clear_user_history,
    count_user_messages_today,
)


@login_required
def home(request: HttpRequest):
    # 🔴 đổi "chat_UI/chat.html" -> "chat.html"
    return render(request, "chat.html", {
        "gemini_ok": is_gemini_configured(),
        "poem_ok": poem_ready(),
    })

# alias route /
def chat_page(request: HttpRequest):
    return home(request)


# --- View chat_api được viết lại hoàn toàn ---
@require_http_methods(["GET", "POST"])
@login_required
def chat_api(request):
    user = request.user
    if request.method == "GET":
        return JsonResponse({"ok": True})

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "invalid json"}, status=400)

    msg = (payload.get("message") or "").strip()
    if not msg:
        return JsonResponse({"ok": False, "error": "empty message"}, status=400)

    # <<< THAY ĐỔI: Kiểm tra quota bằng hàm từ MongoDB >>>
    used_today = count_user_messages_today(user)
    if used_today >= 20:
        return JsonResponse({"ok": False, "error": "quota", "message": "Bạn đã dùng 20 câu hỏi hôm nay."}, status=429)

    # --- Phần xử lý payload giữ nguyên ---
    k = int(payload.get("k") or 5)
    model = payload.get("model") or "gemini-2.5-flash"
    long_answer = bool(payload.get("long_answer"))
    max_tokens = int(payload.get("max_tokens") or 1024)
    # ... (xử lý bullet mode)

    # <<< THAY ĐỔI: Lưu tin nhắn người dùng vào MongoDB >>>
    save_message_to_mongo(user, "user", msg)

    try:
        t0 = now()
        # <<< THAY ĐỔI: Lấy lịch sử từ MongoDB cho bot >>>
        chat_history = get_history_for_bot(user, limit=12)
        
        ret = answer_with_router(
            msg, k=k, gemini_model=model,
            history=chat_history,
            long_answer=long_answer, max_tokens=max_tokens,

        )
        elapsed_ms = (now() - t0).total_seconds() * 1000.0

    except Exception as exc:
        trace = traceback.format_exc()
        logger.exception("chat_api failed")
        error_content = "Xin lỗi, có lỗi kỹ thuật khi xử lý câu hỏi."
        # <<< THAY ĐỔI: Lưu lỗi vào MongoDB >>>
        save_message_to_mongo(user, "assistant", error_content, meta={"error": str(exc), "trace": trace})
        return JsonResponse({"ok": False, "error": "backend", "detail": str(exc)}, status=500)

    answer = ret.get("answer") or "(không có câu trả lời)"
    meta_data = {
        "intent": ret.get("intent"),
        "verification": ret.get("verification"),
        "elapsed_ms": elapsed_ms,
        "error": ret.get("error"),
    }
    
    # <<< THAY ĐỔI: Lưu câu trả lời của bot vào MongoDB >>>
    save_message_to_mongo(user, "assistant", answer, meta=meta_data)
    
    return JsonResponse({
        "ok": True,
        "answer": answer,
        **meta_data
    })

# --- View history_api được viết lại hoàn toàn ---
@login_required
def history_api(request: HttpRequest):
    if request.GET.get("clear") == "1":
        # <<< THAY ĐỔI: Xóa lịch sử trong MongoDB >>>
        clear_user_history(request.user)
        return JsonResponse({"ok": True, "messages": []})

    # <<< THAY ĐỔI: Lấy lịch sử từ MongoDB >>>
    messages = get_history_for_api(request.user)
    return JsonResponse({"ok": True, "messages": messages})

@require_POST
@login_required
def prefs_api(request: HttpRequest):
    """Lưu tuỳ chọn cơ bản (có thể mở rộng sau)."""
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        data = {}
    prof, _ = UserProfile.objects.get_or_create(user=request.user)
    # ví dụ: save_history & daily_limit (tuỳ bạn dùng tới đâu)
    save_history = bool(data.get("save_history", True))
    daily_limit = int(data.get("daily_limit", 20) or 20)
    # bạn có thể thêm field trong model nếu muốn persist 2 giá trị này.
    # Ở bản tối thiểu mình chỉ echo về cho UI.
    return JsonResponse({"ok": True, "save_history": save_history, "daily_limit": daily_limit})
