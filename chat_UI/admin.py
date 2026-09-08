from django.contrib import admin
from django.contrib.auth.models import User
from .models import UserProfile, ChatMessage

admin.site.site_header = "Kiều Bot · Quản trị"
admin.site.site_title = "Kiều Bot Admin"
admin.site.index_title = "Quản lý tài khoản và hội thoại"

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("username", "email", "is_active", "save_history", "daily_quota", "created_at")
    list_filter = ("save_history", "user__is_active", "user__is_staff")
    search_fields = ("user__username", "user__email")
    readonly_fields = ("created_at", "updated_at")
    list_select_related = ("user",)
    actions = ("activate_accounts", "deactivate_accounts")

    @admin.display(ordering="user__username", description="Tên đăng nhập")
    def username(self, obj):
        return obj.user.username

    @admin.display(ordering="user__email", description="Email")
    def email(self, obj):
        return obj.user.email

    @admin.display(boolean=True, ordering="user__is_active", description="Đang hoạt động")
    def is_active(self, obj):
        return obj.user.is_active

    @admin.action(description="Kích hoạt các tài khoản đã chọn")
    def activate_accounts(self, request, queryset):
        User.objects.filter(pk__in=queryset.values_list("user_id", flat=True)).update(is_active=True)

    @admin.action(description="Khóa các tài khoản đã chọn")
    def deactivate_accounts(self, request, queryset):
        User.objects.filter(pk__in=queryset.values_list("user_id", flat=True)).update(is_active=False)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("user", "session_id", "role", "short", "created_at")
    list_filter = ("role", "created_at")
    search_fields = ("content", "session_id", "user__username", "user__email")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at",)
    def short(self, obj): return (obj.content or "")[:60]
