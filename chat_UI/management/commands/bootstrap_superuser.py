"""Safely provision one administrator from deployment-only environment variables."""

import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Create one Django superuser when all DJANGO_SUPERUSER_* variables are set."

    def handle(self, *args, **options):
        credentials = {
            "username": os.getenv("DJANGO_SUPERUSER_USERNAME", "").strip(),
            "email": os.getenv("DJANGO_SUPERUSER_EMAIL", "").strip(),
            "password": os.getenv("DJANGO_SUPERUSER_PASSWORD", ""),
        }
        missing = [name for name, value in credentials.items() if not value]
        if missing:
            self.stdout.write("Superuser bootstrap skipped: credentials are not configured.")
            return

        user_model = get_user_model()
        existing = user_model.objects.filter(username=credentials["username"]).first()
        if existing:
            if existing.is_superuser:
                self.stdout.write("Superuser bootstrap skipped: administrator already exists.")
                return
            raise CommandError("Refusing to replace an existing non-administrator account.")

        if user_model.objects.filter(email__iexact=credentials["email"]).exists():
            raise CommandError("Refusing to reuse an existing account email address.")

        user_model.objects.create_superuser(**credentials)
        self.stdout.write("Superuser created successfully.")
