# Render Deployment

This project is set up to run on Render as two services:

- `daniela-ai-detection` as a static site
- `daniela-bino-backend` as a Python web service

## Backend service

Use the backend from `render.yaml` or configure it manually with:

- Build command: `pip install -r daniela_bino_requirements.txt`
- Start command: `uvicorn daniela_bino_backend:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Recommended environment variables:

- `BINO_CORS_ORIGINS=https://your-frontend.onrender.com`
- `BINO_OBSERVER_MODEL=Qwen/Qwen2.5-0.5B-Instruct`
- `BINO_PERFORMER_MODEL=Qwen/Qwen2.5-0.5B`
- `BINO_THRESHOLD=0.9`
- `BINO_BATCH_SIZE=2`
- `BINO_MAX_LENGTH=256`

Notes:

- The first request can be slow because the backend may need to download and load models.
- This backend is CPU-heavy. Use a Render plan with enough memory for two transformer models.

## Frontend service

The static site publishes the project root. `index.html` redirects to `daniela_ai_detection.html`.

After the backend is deployed, set the backend URL in `render-config.js`:

```js
window.BINO_BACKEND_URL = 'https://your-backend.onrender.com';
```

Then redeploy the static site so the browser uses the deployed API by default.

## Local development

Local development still works with:

- `.\start_daniela_app.bat`

When the page runs on `localhost`, it automatically falls back to `http://127.0.0.1:8008` if no deployed backend URL is configured.
