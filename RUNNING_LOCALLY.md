# Running Daniela AI Detection Locally

## Requirements

- Windows 10 or 11
- [Python 3.10, 3.11, 3.12, or 3.13](https://www.python.org/downloads/) — tick **"Add Python to PATH"** during install
- [Git](https://git-scm.com/download/win)
- ~2 GB free disk space (for the AI models, downloaded automatically on first run)
- ~600 MB RAM free

---

## Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Download **3.10, 3.11, 3.12, or 3.13** — any patch number is fine (e.g. 3.11.0, 3.12.3, 3.13.1)
3. Run the installer
4. **Important:** tick the checkbox "Add Python to PATH" before clicking Install

Verify it worked — open **Command Prompt** and run:
```
python --version
```
You should see something like `Python 3.11.x`, `Python 3.12.x`, or `Python 3.13.x`.

---

## Step 2 — Clone the repository

Open **Command Prompt** (`Win + R` → type `cmd` → Enter) and run:

```
git clone https://github.com/Amasa614/AI-detection.git
cd AI-detection
```

---

## Step 3 — Create a virtual environment

Still in the same Command Prompt window:

```
python -m venv venv
venv\Scripts\activate
```

Your prompt should now show `(venv)` at the start.

---

## Step 4 — Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

This downloads PyTorch (~800 MB) and the other packages. It only needs to run once.

---

## Step 5 — Start the app

```
start_daniela_app.bat
```

This opens two new windows:

| Window | What it does |
|---|---|
| **Daniela Backend** | Runs the FastAPI server on `http://127.0.0.1:8008` |
| **Daniela Frontend** | Serves the HTML on `http://127.0.0.1:8010` |

It also opens the app in your browser automatically.

> **First run only:** the backend will download the AI models (~270 MB total) from
> Hugging Face. This can take 2–5 minutes depending on your internet speed.
> You will see `Application startup complete.` in the Backend window when it is ready.

---

## Step 6 — Use the app

1. Open `http://127.0.0.1:8010/daniela_ai_detection.html` in your browser
2. Paste an essay into the text box
3. Click **Analyse essay** — this runs the fast local analysis (no internet needed)
4. Click **Run Local Binoculars** — this calls the backend LLM (models must be loaded first)

---

## Stopping the app

Close both the **Daniela Backend** and **Daniela Frontend** command windows.

---

## Troubleshooting

### "Python was not found in PATH"
Re-run the Python installer and tick "Add Python to PATH", then restart Command Prompt.

### "Port already in use" error
Another instance is already running. Close all backend/frontend windows, then run:
```
start_daniela_app.bat
```
again.

### Backend window closes immediately
Open Command Prompt, activate venv (`venv\Scripts\activate`), then run manually:
```
python -m uvicorn daniela_bino_backend:app --host 127.0.0.1 --port 8008
```
Read the error message in full — it is usually a missing dependency.

### Models downloading very slowly
The models (~270 MB) download from Hugging Face on first run. If they are slow,
wait — they are cached locally after the first download and never downloaded again.

### "venv\Scripts\activate is not recognized"
Make sure you are in the correct folder:
```
cd YOUR-REPO-NAME
venv\Scripts\activate
```

---

## Running backend and frontend separately

If you only want the backend:
```
start_daniela_bino_backend.bat
```

If you only want to serve the frontend (no binoculars, just local analysis):
```
python -m http.server 8010 --bind 127.0.0.1
```
Then open `http://127.0.0.1:8010/daniela_ai_detection.html`.
