# Deploy Django to Vercel

The production deployment is Django, not Streamlit. Both UIs call the same
`answer_with_router` service, but Django is the single production UI because
it provides login, CSRF protection, quotas and persistent MongoDB history.

Vercel supports Django through its Python runtime and detects a Django project
from `manage.py` and the WSGI application. This repository pins Python 3.12,
uses `project.wsgi:app`, and keeps the Vercel dependency set intentionally
small. The production retriever must be `RETRIEVER=gemini` and
`RERANKER=none`; SentenceTransformer/Jina models do not belong in a serverless
function bundle.

## Vercel environment variables

Set these separately for Preview and Production in the Vercel project:

```text
DJANGO_SECRET_KEY=<long random value>
DEBUG=0
ALLOWED_HOSTS=.vercel.app,your-domain.example
CSRF_TRUSTED_ORIGINS=https://your-domain.example
DATABASE_URL=postgresql://...
MONGO_URI=mongodb+srv://...
MONGO_DB=kieu_bot
MONGO_COL=chunks
MONGO_CHAT_COLLECTION=chat_messages
INDEX_NAME=vector_index
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
RETRIEVER=gemini
GEMINI_EMB_MODEL=models/gemini-embedding-001
RERANKER=none
```

`DATABASE_URL` must point to managed PostgreSQL (for example Vercel Postgres or
Neon). SQLite is only a local fallback and cannot persist accounts/sessions on
serverless instances.

Run migrations against that PostgreSQL database before first production use:

```bash
python manage.py migrate
```

The public readiness endpoint is `/api/health/`. It verifies MongoDB access,
Gemini configuration and the bundled poem corpus without exposing credentials.
