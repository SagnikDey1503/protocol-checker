# Manual Render Deployment Guide

Render does **not** auto-deploy from GitHub for this project. After every code change, redeploy manually from the Render dashboard.

## Your live URLs

| Service | URL |
|---------|-----|
| Frontend | https://protocol-frontend-glbk.onrender.com |
| Backend | https://protocol-backend-glbk.onrender.com |

---

## Step 1 — Redeploy the backend (`protocol-backend`)

1. Open [Render Dashboard](https://dashboard.render.com) → **protocol-backend**
2. Click **Manual Deploy** → **Deploy latest commit**
3. Wait until status is **Live** (first deploy after cold start can take 3–5 minutes)

### Required environment variables (Settings → Environment)

Set these if they are missing:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | From your Render Postgres instance |
| `REDIS_URL` | From your Render Redis/Key Value instance |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `GOOGLE_API_KEY` | Your Google AI Studio key |
| `GROQ_API_KEY` | Your Groq API key |
| `JWT_SECRET_KEY` | Any long random string (keep stable across deploys) |
| `CORS_ORIGINS` | `https://protocol-frontend-glbk.onrender.com` |
| `LLM_PROVIDER` | `gemini` |
| `FAST_LLM_PROVIDER` | `groq` |
| `DEBUG` | `false` |
| `LOG_LEVEL` | `INFO` |

**Important:** `CORS_ORIGINS` must exactly match your frontend URL (no trailing slash).

### Verify backend is up

```bash
curl https://protocol-backend-glbk.onrender.com/api/v1/health
```

You should get JSON like `{"status":"healthy",...}`. If you see `Not Found` with no JSON, the backend service is not running — check Render logs.

---

## Step 2 — Redeploy the frontend (`protocol-frontend`)

1. Open **protocol-frontend** in Render
2. Settings → Environment → confirm:

| Key | Value |
|-----|-------|
| `VITE_API_BASE_URL` | `https://protocol-backend-glbk.onrender.com` |

3. Click **Manual Deploy** → **Deploy latest commit**

The frontend bakes `VITE_API_BASE_URL` in at build time, so you must redeploy after changing it.

---

## Step 3 — Test login and PDF upload

1. Open https://protocol-frontend-glbk.onrender.com
2. You should see the **Sign In / Create Account** screen
3. Click **Create one here** → register with email + password (min 8 chars)
4. After login, click **Upload Protocol** and upload a PDF

---

## Common errors

### "Failed to fetch" / CORS if backend is down

The backend service is not running. Redeploy `protocol-backend` and check logs for startup errors (missing API keys, DB connection, etc.).

### CORS error in browser console

Set `CORS_ORIGINS=https://protocol-frontend-glbk.onrender.com` on the backend and redeploy.

### 401 on upload / protocol list

Log out and sign in again. Auth is now required — the old mock token no longer works.

### Backend OOM on free tier

The Dockerfile uses 1 uvicorn worker. If deploys fail during model load, check Render logs for memory errors.

---

## Local development

```bash
# Backend
cp .env.example .env   # fill in API keys
docker compose up -d postgres redis
source .venv/bin/activate
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — backend CORS allows `http://localhost:5173` by default.
