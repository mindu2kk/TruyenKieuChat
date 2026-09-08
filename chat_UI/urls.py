from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_page, name="chat"),
    path("api/chat/", views.chat_api, name="chat_api"),
    path("api/history/", views.history_api, name="history_api"),
    path("api/prefs/", views.prefs_api, name="prefs_api"),
    path("api/health/", views.health_api, name="health_api"),
    # Preserve existing clients that called the original no-slash endpoints.
    path("api/chat", views.chat_api, name="chat_api_legacy"),
    path("api/history", views.history_api, name="history_api_legacy"),
    path("api/prefs", views.prefs_api, name="prefs_api_legacy"),
]
