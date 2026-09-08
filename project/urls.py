from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),  # login Google
    path("", include("chat_UI.urls")),            # trang chat
]
