# app.py
import os, time, json, logging, uuid
from functools import wraps
from typing import Any, Dict
from dataclasses import dataclass

import requests
from flask import Flask, request, jsonify, g

# Using .env file for environment variables and configs
from dotenv import load_dotenv
load_dotenv()

# ========= Config =========
@dataclass
class Config:
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Server-side API key to authenticate incoming requests to *your* service
    SERVER_API_KEY: str = os.getenv("SERVER_API_KEY", "")

    # Upstream LLM config
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds

cfg = Config()

# ========= Logging =========
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(g, "request_id", None),
            "path": getattr(request, "path", None),
            "method": getattr(request, "method", None),
            "remote_addr": request.headers.get("X-Forwarded-For", request.remote_addr) if request else None,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

logger = logging.getLogger("llm_api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)

# ========= Flask App =========
app = Flask(__name__)

@app.before_request
def inject_request_id():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

def require_server_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        expected = cfg.SERVER_API_KEY
        provided = request.headers.get("X-API-Key", "")
        if expected and provided != expected:
            logger.warning("Unauthorized request")
            return error_response(401, "Unauthorized")
        return f(*args, **kwargs)
    return wrapper

# ========= Error helpers =========
def error_response(status: int, message: str, detail: Dict[str, Any] | None = None):
    resp = {
        "error": {
            "message": message,
            "status": status,
            "request_id": g.get("request_id"),
            "detail": detail or {},
        }
    }
    return jsonify(resp), status

@app.errorhandler(400)
def bad_request(e): return error_response(400, "Bad Request")
@app.errorhandler(404)
def not_found(e): return error_response(404, "Not Found")
@app.errorhandler(405)
def method_not_allowed(e): return error_response(405, "Method Not Allowed")
@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled exception")
    return error_response(500, "Internal Server Error")

# ========= Validation =========
def validate_generate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("JSON body required")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Field 'prompt' (non-empty string) is required")
    max_tokens = payload.get("max_tokens", 256)
    if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > 8192:
        raise ValueError("Field 'max_tokens' must be an int in (0, 8192]")
    temperature = payload.get("temperature", 0.2)
    if not isinstance(temperature, (int, float)) or not (0 <= float(temperature) <= 2):
        raise ValueError("Field 'temperature' must be between 0 and 2")
    # passthrough extras
    return {"prompt": prompt, "max_tokens": max_tokens, "temperature": float(temperature)}

# ========= Upstream LLM call =========
def call_llm_chat_completions(prompt: str, max_tokens: int, temperature: float) -> str:
    """
    Uses a broadly compatible 'chat/completions' schema.
    Works with OpenAI-compatible endpoints and many open-source gateways.
    """
    url = f"{cfg.LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        # Debugging
        # logger.info({
        #     "event": "llm_request_debug",
        #     "url": url,
        #     "has_auth": bool(headers.get("Authorization")),
        #     "auth_prefix": headers.get("Authorization", "")[:10],  # "Bearer sk-"
        #     "model": body.get("model"),
        #     "base": cfg.LLM_API_BASE,
        # })

        r = requests.post(url, headers=headers, json=body, timeout=cfg.LLM_TIMEOUT)

        # logger.info({
        #     "event": "llm_response_debug",
        #     "status": r.status_code,
        #     "text_head": r.text[:200]
        # })

    except requests.Timeout as e:
        logger.warning("LLM timeout")
        raise UpstreamError(504, "Upstream LLM timed out") from e
    except requests.RequestException as e:
        logger.exception("LLM request failed")
        raise UpstreamError(502, "Upstream LLM request failed") from e

    if r.status_code == 401:
        logger.error("LLM unauthorized")
        raise UpstreamError(502, "LLM credentials invalid")
    if r.status_code >= 400:
        detail = safe_json(r)
        logger.error(f"LLM error: {r.status_code} {detail}")
        raise UpstreamError(502, "LLM returned an error", detail)

    data = safe_json(r)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Unexpected LLM schema")
        raise UpstreamError(502, "Unexpected LLM response schema", {"raw": data}) from e

def safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}

# Class for errors with model provider
class UpstreamError(RuntimeError):
    def __init__(self, status: int, msg: str, detail: Dict[str, Any] | None = None):
        super().__init__(msg)
        self.status = status
        self.detail = detail or {}

# ========= Routes =========
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True, "ts": int(time.time()), "request_id": g.request_id})

@app.route("/v1/generate", methods=["POST"])
@require_server_key
def generate():
    start = time.time()
    try:
        payload = validate_generate_payload(request.get_json(silent=True))
    except ValueError as ve:
        return error_response(400, str(ve))

    logger.info(f"Generate request received")

    try:
        output = call_llm_chat_completions(
            prompt=payload["prompt"],
            max_tokens=payload["max_tokens"],
            temperature=payload["temperature"],
        )
    except UpstreamError as ue:
        return error_response(ue.status, str(ue), ue.detail)

    latency_ms = int((time.time() - start) * 1000)
    logger.info(f"Generate success | latency_ms={latency_ms}")
    return jsonify({
        "request_id": g.request_id,
        "latency_ms": latency_ms,
        "model": cfg.LLM_MODEL,
        "output": output,
    })

if __name__ == "__main__":
    app.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
