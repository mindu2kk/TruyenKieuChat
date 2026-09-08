from django.contrib import admin
from .models import UserProfile, ChatMessage

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "save_history", "daily_quota")
    search_fields = ("user__username", "user__email")

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("user", "session_id", "role", "short", "created_at")
    list_filter = ("role",)
    search_fields = ("content", "session_id", "user__username", "user__email")
    def short(self, obj): return (obj.content or "")[:60]
