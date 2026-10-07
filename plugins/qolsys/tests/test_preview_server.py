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


def h264(*nal_types):
    return b"".join(b"\x00\x00\x00\x01" + bytes([t]) + b"data" for t in nal_types)


@pytest.mark.asyncio
async def test_chaining_repeats_segments_until_view_limit():
    calls = []

    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            calls.append(seconds)
            await emit(h264(5, 1))
            await asyncio.sleep(0.02)
            return parameters

    parameters = SimpleNamespace(parameter_sets=(b"\x67sps", b"\x68pps"))
    session = PreviewServer(Client(), parameters, max_view_seconds=0.1)
    port = int((await session.start()).rsplit(":", 1)[1])
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = await asyncio.wait_for(reader.read(), 2)
    await session.finished.wait()
    assert len(calls) >= 3 and session.error is None
    assert data.count(h264(5, 1)) >= 3
    writer.close()
    await writer.wait_closed()
    await session.close()


@pytest.mark.asyncio
async def test_viewer_disconnect_stops_chaining():
    calls = []

    class Client:
        async def capture_preview(self, seconds, emit, parameters):
            calls.append(seconds)
            await asyncio.sleep(0.05)
            return parameters

    session = PreviewServer(Client(), None, max_view_seconds=60)
    port = int((await session.start()).rsplit(":", 1)[1])
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    await asyncio.sleep(0.12)
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(session.finished.wait(), 2)
    count = len(calls)
    await asyncio.sleep(0.15)
    assert len(calls) == count  # no new segments after the viewer left
    await session.close()



SAMPLE = None


def sample_h264(tmp_path):
    """Synthetic 1280x720 H264 at a panel-like bitrate, so no camera footage is involved."""
    import subprocess
    out = tmp_path / "sample.h264"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=10", "-t", "3", "-c:v", "libx264", "-profile:v", "baseline",
                    "-bf", "0", "-b:v", "4M", "-pix_fmt", "yuv420p", "-f", "h264", str(out)], check=True)
    return out.read_bytes()


@pytest.mark.asyncio
@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")
async def test_frame_holder_turns_bursts_into_steady_decodable_stream(tmp_path):
    import itertools

    from frame_holder import FrameHolder
    data = sample_h264(tmp_path)
    arrivals, out = [], bytearray()
    loop = asyncio.get_running_loop()
    start = loop.time()

    async def output(chunk):
        arrivals.append(loop.time() - start)
        out.extend(chunk)

    holder = FrameHolder("ffmpeg", output, fps=10)
    await holder.start()
    await holder.feed(data[: len(data) // 3])
    await asyncio.sleep(1.5)  # panel-style silence
    await holder.feed(data[len(data) // 3:])
    await asyncio.sleep(1.0)
    await holder.close()
    gaps = [b - a for a, b in itertools.pairwise(arrivals)]
    assert arrivals and max(gaps) < 0.3  # steady output through the silence
    probe = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=nb_read_frames", "-of", "csv=p=0", "-", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    frames, _ = await probe.communicate(bytes(out))
    assert int(frames.strip()) >= 15


@pytest.mark.asyncio
@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")
async def test_frame_holder_processes_end_with_the_viewer(tmp_path):
    from frame_holder import FrameHolder

    async def output(chunk):
        pass

    holder = FrameHolder("ffmpeg", output, fps=10)
    await holder.start()
    await holder.feed(sample_h264(tmp_path)[:20000])
    await holder.close()
    assert holder.decoder.returncode is not None and holder.encoder.returncode is not None


@pytest.mark.asyncio
@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="needs ffmpeg")
async def test_holder_process_writes_steady_stream_to_viewer_socket(tmp_path):
    import socket

    from frame_holder import HolderProcess
    data = sample_h264(tmp_path)
    ours, viewer = socket.socketpair()
    reader, _ = await asyncio.open_connection(sock=viewer)
    holder = HolderProcess("ffmpeg", ours, fps=10, log=lambda line: None)
    await holder.start()
    await holder.feed(data[: len(data) // 2])
    await asyncio.sleep(1.5)  # panel-style silence
    await holder.feed(data[len(data) // 2:])
    received = bytearray()
    loop = asyncio.get_running_loop()
    end = loop.time() + 1.0
    while loop.time() < end:
        try:
            received.extend(await asyncio.wait_for(reader.read(65536), 0.3))
        except TimeoutError:
            break
    await asyncio.wait_for(holder.close(), 10)  # never hangs on close
    ours.close()
    assert holder.process.returncode is not None
    assert len(received) > 10_000
