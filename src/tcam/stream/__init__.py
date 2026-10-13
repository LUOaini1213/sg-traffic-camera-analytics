"""Kafka-API streaming: frames topic -> detector -> camera-state topic -> HTTP API."""
import os

FRAMES = "tcam.frames"
STATE = "tcam.camera-state"
BOOTSTRAP = os.environ.get("TCAM_BOOTSTRAP", "127.0.0.1:39092")
