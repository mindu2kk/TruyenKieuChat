"""Deterministic settings for the automated test harness."""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-secret-key-not-for-production")

from .settings import *  # noqa: E402,F403


# Tests render templates directly without running collectstatic first. Production
# keeps the hashed, compressed manifest storage configured in ``settings.py``.
STORAGES = {  # noqa: F405
    **STORAGES,
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}

# The Django test client uses plain HTTP by default. HTTPS behavior is covered by
# the production settings check instead of turning every test response into 301.
SECURE_SSL_REDIRECT = False
