from django.apps import AppConfig

class ChatUIConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat_UI"

    def ready(self):
        # Register lifecycle hooks only after Django's app registry is ready.
        from . import signals  # noqa: F401
