from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_page, name="chat"),
    path("api/chat/", views.chat_api, name="chat_api"),
    path("api/history/", views.history_api, name="history_api"),
    path("api/conversations/", views.conversations_api, name="conversations_api"),
    path("api/conversations/<str:conversation_id>/", views.conversation_detail_api, name="conversation_detail_api"),
    path("api/messages/<str:message_id>/actions/", views.message_actions_api, name="message_actions_api"),
    path("api/prefs/", views.prefs_api, name="prefs_api"),
    path("api/health/", views.health_api, name="health_api"),
    # Preserve existing clients that called the original no-slash endpoints.
    path("api/chat", views.chat_api, name="chat_api_legacy"),
    path("api/history", views.history_api, name="history_api_legacy"),
    path("api/prefs", views.prefs_api, name="prefs_api_legacy"),
]
