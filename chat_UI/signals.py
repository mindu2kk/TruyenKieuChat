"""Keep a controllable profile record for every Django account."""

from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import UserProfile


@receiver(post_save, sender=User)
def ensure_user_profile(sender, instance: User, **kwargs) -> None:
    """Create the management profile exactly once, including for social users."""
    UserProfile.objects.get_or_create(user=instance)
