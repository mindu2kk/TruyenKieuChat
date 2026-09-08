"""Application package bootstrap."""

from pathlib import Path

from dotenv import load_dotenv

# All entry points (Django, Streamlit, tests and management commands) resolve
# the same local development configuration. Production platforms inject these
# variables and therefore take precedence over a local .env file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
