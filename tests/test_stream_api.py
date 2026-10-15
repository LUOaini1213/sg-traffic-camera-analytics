import json

import pytest

pytest.importorskip("fastapi")
from tcam.stream import api  # noqa: E402


@pytest.fixture(autouse=True)
def empty_view():
    api.LATEST.clear()
    yield
    api.LATEST.clear()


def test_valid_state_is_kept():
    st = {"camera": "1701", "counts": {"total": 12}, "congested": False, "capture": "2026-10-14T08:00"}
    assert api.apply_state(json.dumps(st).encode())
    assert api.LATEST["1701"] == st


@pytest.mark.parametrize("raw", [
    b"not json", b"\xff\xfe", b"[]", b"null", b'"1701"', b'{"counts": {"total": 1}, "congested": false}',
    b'{"camera": "1701", "congested": false}', b'{"camera": "1701", "counts": "12", "congested": false}',
    b'{"camera": "1701", "counts": {"total": "many"}, "congested": false}',
    b'{"camera": 1701, "counts": {"total": 1}, "congested": false}',
])
def test_malformed_state_is_skipped_without_raising(raw):
    assert api.apply_state(raw) is False
    assert api.LATEST == {}


def test_responses_carry_security_headers():
    from fastapi.testclient import TestClient
    client = TestClient(api.app)  # no `with`: the lifespan (Kafka consumer) does not start
    for r in (client.get("/health"), client.get("/cameras/none")):  # 200 and 404
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
