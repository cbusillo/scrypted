"""Verify actual local TCP delivery and producer cleanup on consumer close."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from preview_server import PreviewServer


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_operation", ["drain", "close"])
async def test_slow_consumer_does_not_leave_preview_busy(monkeypatch, blocked_operation):
    monkeypatch.setattr("preview_server.DRAIN_TIMEOUT", 0.01)
    monkeypatch.setattr("preview_server.CLOSE_TIMEOUT", 0.01)
    cleaned = asyncio.Event()

    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            try:
                await emit(b"new frame")
                return parameters
            finally:
                cleaned.set()

    class Reader:
        async def read(self, size):
            await asyncio.Future()

    class Writer:
        transport = SimpleNamespace(abort=Mock())
        write = Mock()
        close = Mock()

        async def drain(self):
            if blocked_operation == "drain":
                await asyncio.Future()

        async def wait_closed(self):
            if blocked_operation == "close":
                await asyncio.Future()

    session = PreviewServer(Client(), "parameters")
    session.server = SimpleNamespace(close=Mock())
    writer = Writer()
    await asyncio.wait_for(session._handle(Reader(), writer), 1)
    assert cleaned.is_set() and session.finished.is_set()
    if blocked_operation == "drain":
        assert isinstance(session.error, TimeoutError)
    else:
        writer.transport.abort.assert_called_once()


@pytest.mark.asyncio
async def test_tcp_delivers_new_packets_and_finishes():
    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            await emit(b"first")
            await emit(b"second")
            return parameters
    session = PreviewServer(Client(), "parameters")
    url = await session.start()
    port = int(url.rsplit(":", 1)[1])
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    assert await reader.read() == b"firstsecond"
    await session.finished.wait()
    assert session.error is None
    writer.close()
    await writer.wait_closed()
    await session.close()


@pytest.mark.asyncio
async def test_tcp_disconnect_cancels_producer_and_waits_for_cleanup():
    started, cleaned = asyncio.Event(), asyncio.Event()
    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            try:
                started.set()
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.01)
                cleaned.set()
    session = PreviewServer(Client(), None)
    port = int((await session.start()).rsplit(":", 1)[1])
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    await started.wait()
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(session.finished.wait(), 2)
    assert cleaned.is_set()
    await session.close()


@pytest.mark.asyncio
async def test_explicit_server_close_waits_for_producer_cleanup():
    started, cleaned = asyncio.Event(), asyncio.Event()
    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            try:
                started.set()
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.01)
                cleaned.set()
    session = PreviewServer(Client(), None)
    port = int((await session.start()).rsplit(":", 1)[1])
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    await started.wait()
    await session.close()
    assert cleaned.is_set() and session.finished.is_set()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_reader_waiting_for_socket_close_does_not_deadlock_cleanup():
    closed, reader_finished = asyncio.Event(), asyncio.Event()

    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            return parameters

    class Reader:
        async def read(self, size):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await closed.wait()
            finally:
                reader_finished.set()

    class Writer:
        transport = SimpleNamespace(abort=Mock())

        def close(self):
            closed.set()

        async def wait_closed(self):
            await reader_finished.wait()

    session = PreviewServer(Client(), "parameters")
    session.server = SimpleNamespace(close=Mock())
    await asyncio.wait_for(session._handle(Reader(), Writer()), 1)
    assert reader_finished.is_set() and session.finished.is_set()
