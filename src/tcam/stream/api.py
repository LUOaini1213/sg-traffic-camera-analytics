"""Read-only HTTP view of the latest state per camera (FastAPI).

Run: uvicorn tcam.stream.api:app --port 8088
"""
from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from . import BOOTSTRAP, STATE

LATEST: dict[str, dict] = {}


def apply_state(raw: bytes | str) -> bool:
    """Fold one camera-state message into LATEST. A malformed message is skipped and reported as False:
    an exception here would end the consumer thread and leave the API serving a frozen view."""
    try:
        st = json.loads(raw)
        cam = st["camera"]
        int(st["counts"]["total"]), bool(st["congested"])  # the fields /cameras reads
    except (ValueError, TypeError, KeyError):
        return False
    if not isinstance(cam, str):
        return False
    LATEST[cam] = st
    return True


def _consume(bootstrap: str = BOOTSTRAP) -> None:
    from confluent_kafka import Consumer
    # a fresh group each start: the API rebuilds its view from the start of the state topic
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"tcam-api-{uuid.uuid4().hex[:8]}",
                  "auto.offset.reset": "earliest", "enable.auto.commit": False})
    c.subscribe([STATE])
    while True:
        m = c.poll(1.0)
        if m is None or m.error():
            continue
        apply_state(m.value())


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_consume, daemon=True).start()
    yield


app = FastAPI(title="sg-traffic-camera-analytics", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """JSON only, so no content may load; the Swagger and ReDoc pages keep FastAPI's defaults."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    if not request.url.path.startswith(("/docs", "/redoc")):
        response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    return response


@app.get("/health")
def health() -> dict:
    return {"cameras": len(LATEST)}


@app.get("/cameras")
def cameras() -> list[dict]:
    return [{"camera": k, "total": v["counts"]["total"], "congested": v["congested"], "capture": v.get("capture")}
            for k, v in sorted(LATEST.items())]


@app.get("/cameras/{cam}")
def camera(cam: str) -> dict:
    if cam not in LATEST:
        raise HTTPException(404, "unknown camera")
    return LATEST[cam]
