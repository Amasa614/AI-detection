# Hugging Face Endpoint Deployment

This project can be deployed with:

- frontend on Render Static Sites
- binoculars backend on a Hugging Face custom container Inference Endpoint

## Why this setup

The backend computes binoculars-style scores from full token distributions of two models. That requires direct access to model logits, so a basic hosted text-generation API is not enough. A custom container endpoint keeps the current `POST /score` API intact.

## Files used for the endpoint

- `daniela_bino_backend.py`
- `requirements.txt`
- `Dockerfile`
- `hf-start.sh`

## Create the Hugging Face endpoint

Create an Inference Endpoint using a custom container image built from this repo.

Use a machine type with enough memory for two transformer models. CPU may still work for smaller tests, but a GPU-backed endpoint is the safer production choice.

## Required environment variables

- `BINO_CORS_ORIGINS=https://your-frontend.onrender.com`
- `BINO_PRELOAD_MODELS=true`

Optional environment variables:

- `BINO_OBSERVER_MODEL=Qwen/Qwen2.5-0.5B-Instruct`
- `BINO_PERFORMER_MODEL=Qwen/Qwen2.5-0.5B`
- `BINO_THRESHOLD=0.9`
- `BINO_BATCH_SIZE=2`
- `BINO_MAX_LENGTH=256`
- `BINO_MIN_SENTENCE_WORDS=5`

If you want to mount or point to alternate local model paths inside the container:

- `BINO_OBSERVER_MODEL_PATH=/repository/observer`
- `BINO_PERFORMER_MODEL_PATH=/repository/performer`

Model path variables take precedence over model IDs.

## Endpoint behavior

The container exposes:

- `GET /`
- `GET /health`
- `POST /score`

The frontend expects `POST /score` on the endpoint root URL.

## Frontend wiring

Once Hugging Face gives you an endpoint URL, update `render-config.js`:

```js
window.BINO_BACKEND_URL = 'https://your-endpoint.endpoints.huggingface.cloud';
```

Then redeploy the frontend static site.

## Local development

Local development still works with:

- `.\start_daniela_app.bat`

That script resets `render-config.js` to local mode and uses `http://127.0.0.1:8008`.

## Notes

- The first startup may still be slow while models download and warm up.
- `/health` reports whether models are loaded and surfaces any load error.
- If you keep the frontend on a different domain, CORS must include that frontend origin.
