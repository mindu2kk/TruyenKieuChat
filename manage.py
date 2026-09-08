#!/usr/bin/env python
# manage.py — entrypoint for the Django project
import os
import sys


def main() -> None:
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it's installed and available on your "
            "PYTHONPATH environment variable. You can install it with:\n\n"
            "    pip install django\n"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
