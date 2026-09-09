FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY --chown=app:app manage.py ./
COPY --chown=app:app project ./project
COPY --chown=app:app chat_UI ./chat_UI
COPY --chown=app:app app ./app
COPY --chown=app:app data/interim/poem ./data/interim/poem

RUN chown app:app /app
USER app

# Collecting static assets needs Django settings but never a production secret.
RUN DJANGO_SECRET_KEY=container-build-only-not-a-runtime-secret-1234567890 DEBUG=0 \
    python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["sh", "-c", "exec gunicorn project.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-120} --access-logfile - --error-logfile -"]
