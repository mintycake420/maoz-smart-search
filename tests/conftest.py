from __future__ import annotations

import socket

import pytest

from maoz_search import SearchEngine


@pytest.fixture(scope="session")
def engine() -> SearchEngine:
    """Load and warm the real local encoder while outbound connections are blocked."""

    search_engine = SearchEngine.from_default()
    original_create_connection = socket.create_connection

    def blocked(*_args, **_kwargs):
        raise AssertionError("Runtime attempted a network connection")

    socket.create_connection = blocked
    try:
        search_engine.encoder.encode(["בדיקת קידוד מקומית"])
    finally:
        socket.create_connection = original_create_connection
    return search_engine
