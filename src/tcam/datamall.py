"""LTA DataMall client.

The AccountKey is read from the LTA_KEY environment variable at run time only.
It is never logged, printed or written to disk by this package.
"""
from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://datamall2.mytransport.sg/ltaodataservice"
UA = "sg-traffic-camera-analytics/0.1"


def _headers() -> dict:
    key = os.environ.get("LTA_KEY")
    if not key:
        raise RuntimeError("set LTA_KEY (DataMall AccountKey) in the environment")
    return {"AccountKey": key, "accept": "application/json", "User-Agent": UA}


def get_json(endpoint: str, skip: int = 0, timeout: float = 30) -> list[dict]:
    url = f"{BASE}/{endpoint}" + (f"?$skip={skip}" if skip else "")
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["value"]


def traffic_images() -> list[dict]:
    """CameraID, Latitude, Longitude, ImageLink (signed link, expires after ~5 minutes)."""
    return get_json("Traffic-Imagesv2")


def traffic_incidents() -> list[dict]:
    return get_json("TrafficIncidents")


def speed_bands(workers: int = 8, page: int = 500) -> list[dict]:
    """All TrafficSpeedBands records (about 144k links, ~290 pages), fetched in parallel."""
    first = get_json("v4/TrafficSpeedBands")
    out = list(first)
    if len(first) < page:
        return out
    skip = page
    with ThreadPoolExecutor(workers) as ex:
        while True:
            skips = [skip + i * page for i in range(workers * 4)]
            pages = list(ex.map(lambda s: get_json("v4/TrafficSpeedBands", s), skips))
            done = False
            for p in pages:
                out.extend(p)
                if len(p) < page:
                    done = True
            if done:
                return out
            skip = skips[-1] + page


def download(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()
