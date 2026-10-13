"""Read-only HTTP view of the latest state per camera (FastAPI).

Run: uvicorn tcam.stream.api:app --port 8088
"""
from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import BOOTSTRAP, STATE

LATEST: dict[str, dict] = {}


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
        st = json.loads(m.value())
        LATEST[st["camera"]] = st


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_consume, daemon=True).start()
    yield


app = FastAPI(title="sg-traffic-camera-analytics", lifespan=lifespan)


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
