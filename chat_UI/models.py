from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    save_history = models.BooleanField(default=True)
    daily_quota = models.PositiveIntegerField(default=20)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile({self.user})"

class ChatMessage(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    session_id = models.CharField(max_length=64, db_index=True)
    role = models.CharField(max_length=16, choices=[("user","user"),("assistant","assistant")])
    content = models.TextField()
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["session_id", "created_at"]),
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        u = self.user.username if self.user_id else "anon"
        return f"{u}/{self.session_id} {self.role}: {self.content[:40]}..."
