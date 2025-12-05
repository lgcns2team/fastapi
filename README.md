# fastapi

Simple FastAPI project — Bedrock Gateway API (local development instructions)

## Requirements
- Python 3.11

## Setup (macOS / zsh)
1. Create a virtual environment using Python 3.11:

```bash
cd /Users/sung0/Desktop/Hai_project.ai/fastapi
python3.11 -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Upgrade packaging tools and install dependencies:

```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Run the development server

You can run using `uvicorn` (recommended for development with auto-reload):

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Or using `python` directly:

```bash
source .venv/bin/activate
python main.py
```

By default the app will listen on `127.0.0.1:8000`. Visit `http://127.0.0.1:8000/docs` for the OpenAPI UI.

## Notes
- The project expects AWS credentials for Bedrock-related calls if you use the Bedrock endpoints. For local dev, either set `AWS_REGION` and credentials in environment variables or mock the client.
- Add `.venv/` to your `.gitignore` (already included).
