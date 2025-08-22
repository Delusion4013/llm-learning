## Summary

This is a simple Flask backend hosting a LLM api endpoint.

## Setup

Using [uv](https://docs.astral.sh/uv/) to quickly setup virtual environment.

```bash
# At toy_projects folder
uv init flask_backend
cd flask_backend
uv add requests flask python-dotenv
```

Create a `.env` file with the following content:

```ini
SERVER_API_KEY=change-me        
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
PORT=8000
DEBUG=false
```

## Run

#### Start server

```bash
uv run app.py
```

#### Check server status

```bash
curl http://localhost:8000/healthz
```

You should get something like:
```json
{
  "ok": true,
  "request_id": "1247529b-f6cd-47cb-8675-922b667dea19",
  "ts": 1755831779
}
```


#### Sending request

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"ping","max_tokens":5,"temperature":0.2}'
```

A sample successful response with commandline logging output:

```json
{
  "latency_ms": 2319,
  "model": "gpt-4o-mini",
  "output": "Pong! How can",
  "request_id": "f32ba901-97e5-4a3d-9582-4b3e8b6228b1"
}
```


A sample unsuccessful response:
```json
{
  "error": {
    "detail": {},
    "message": "LLM credentials invalid",
    "request_id": "aa134b5c-ea32-4e7b-9a9e-f3ea8da9e3a5",
    "status": 502
  }
}
```