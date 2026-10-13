"""data.gov.sg archive of the same LTA traffic-camera feed.

DataMall only serves the current image. The data.gov.sg endpoint accepts `date_time` and
returns the frame that was current at that time, with a persistent image URL and MD5.
Until 2026-06-29 it listed 90 cameras; from 2026-06-30 it lists 8 (checked 2026-10-13).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request

API = "https://api.data.gov.sg/v1"
UA = {"User-Agent": "sg-traffic-camera-analytics/0.1"}


def _get(url: str, timeout: float = 30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.load(r)


def cameras_at(dt: str) -> list[dict]:
    """dt like '2026-01-15T12:00:00' (Singapore time)."""
    d = _get(f"{API}/transport/traffic-images?date_time={dt}")
    return d["items"][0]["cameras"] if d.get("items") else []


def rainfall_at(dt: str) -> tuple[list[dict], list[dict]]:
    d = _get(f"{API}/environment/rainfall?date_time={dt}")
    readings = d["items"][0]["readings"] if d.get("items") else []
    return d["metadata"]["stations"], readings


def fetch_image(url: str, md5: str | None = None, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        b = r.read()
    if md5 and hashlib.md5(b).hexdigest() != md5:
        raise ValueError(f"md5 mismatch for {url}")
    return b
