"""Turn bursty panel H264 into a steady stream that viewers can pace.

The panel only offers whole-file downloads of a growing recording, so frames
arrive in clumps with silent gaps. Viewers such as Apple Home read silence as
a failing network. FrameHolder decodes the bursty input, keeps the newest
picture, and re-encodes exactly `fps` pictures per second, repeating the last
picture through gaps.

The plugin runs it in its own process (HolderProcess) so panel downloads in
the plugin's event loop cannot starve the steady clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable

WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2  # yuv420p


class FrameHolder:
    def __init__(self, ffmpeg: str, output: Callable[[bytes], Awaitable[None]], fps: int = 10,
                 initial_jpeg: bytes | None = None, log: Callable[[str], None] = print):
        self.ffmpeg = ffmpeg
        self.output = output
        self.fps = fps
        self.initial_jpeg = initial_jpeg
        self.latest: bytes | None = None
        self.decoder: asyncio.subprocess.Process | None = None
        self.encoder: asyncio.subprocess.Process | None = None
        self.tasks: list[asyncio.Task] = []
        self.log = log
        self.decoded = 0
        self.sent = 0
        self.started = time.monotonic()

    async def start(self):
        if self.initial_jpeg:
            self.latest = await self._decode_jpeg(self.initial_jpeg)
        self.decoder = await asyncio.create_subprocess_exec(
            # analyzeduration 0 means "default" (seconds of probing); use the
            # smallest probe and one thread so frames leave as they arrive.
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-probesize", "32",
            "-analyzeduration", "1", "-fpsprobesize", "0", "-framerate", str(self.fps), "-threads", "1",
            "-f", "h264", "-i", "pipe:0",
            "-vf", f"scale={WIDTH}:{HEIGHT}", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.encoder = await asyncio.create_subprocess_exec(
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(self.fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline",
            "-g", str(self.fps), "-bf", "0", "-flush_packets", "1", "-f", "h264", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.tasks = [asyncio.create_task(self._read_decoded()), asyncio.create_task(self._tick()),
                      asyncio.create_task(self._forward()),
                      asyncio.create_task(self._log_stderr("decoder", self.decoder)),
                      asyncio.create_task(self._log_stderr("encoder", self.encoder))]
        for task in self.tasks:
            task.add_done_callback(self._report)
        self.log(f"frame holder started; opening picture {'ready' if self.latest else 'missing'}")

    def _report(self, task: asyncio.Task):
        if not task.cancelled() and task.exception():
            self.log(f"frame holder {task.get_coro().__qualname__} stopped: {task.exception()!r} "
                     f"after {time.monotonic() - self.started:.1f}s, {self.decoded} decoded, {self.sent} sent")

    async def _log_stderr(self, name: str, process: asyncio.subprocess.Process):
        assert process.stderr
        while line := await process.stderr.readline():
            self.log(f"frame holder {name}: {line.decode(errors='replace').rstrip()}")

    async def _decode_jpeg(self, jpeg: bytes) -> bytes | None:
        process = await asyncio.create_subprocess_exec(
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "image2pipe", "-i", "pipe:0",
            "-vf", f"scale={WIDTH}:{HEIGHT}", "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        raw, _ = await process.communicate(jpeg)
        return raw if len(raw) == FRAME_BYTES else None

    async def feed(self, h264: bytes):
        """Accept new Annex B H264 from the panel, in whatever bursts it arrives."""
        if self.decoder and self.decoder.stdin and not self.decoder.stdin.is_closing():
            self.decoder.stdin.write(h264)
            await self.decoder.stdin.drain()

    async def _read_decoded(self):
        assert self.decoder and self.decoder.stdout
        while True:
            self.latest = await self.decoder.stdout.readexactly(FRAME_BYTES)
            self.decoded += 1

    async def _tick(self):
        # A steady clock, not input arrival, decides when pictures go out.
        assert self.encoder and self.encoder.stdin
        loop = asyncio.get_running_loop()
        interval = 1 / self.fps
        next_time = loop.time()
        while True:
            if self.latest is not None:
                self.encoder.stdin.write(self.latest)
                await self.encoder.stdin.drain()
                self.sent += 1
            next_time += interval
            await asyncio.sleep(max(0.0, next_time - loop.time()))

    async def _forward(self):
        assert self.encoder and self.encoder.stdout
        while chunk := await self.encoder.stdout.read(65536):
            await self.output(chunk)

    async def close(self):
        self.log(f"frame holder closing after {time.monotonic() - self.started:.1f}s: "
                 f"{self.decoded} live frames decoded, {self.sent} pictures sent")
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for process in (self.decoder, self.encoder):
            if not process:
                continue
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            # Since Python 3.12, wait() also waits for the output pipes to close,
            # so drain them; an unread full pipe would block forever.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.communicate(), 5)


class HolderProcess:
    """Run FrameHolder in a child process that writes straight to the viewer's socket."""

    def __init__(self, ffmpeg: str, sock: socket.socket, fps: int = 10,
                 initial_jpeg: bytes | None = None, log: Callable[[str], None] = print):
        self.ffmpeg = ffmpeg
        self.sock = sock
        self.fps = fps
        self.initial_jpeg = initial_jpeg
        self.log = log
        self.process: asyncio.subprocess.Process | None = None
        self.jpeg_path: str | None = None
        self.tasks: list[asyncio.Task] = []

    async def start(self):
        fd = self.sock.fileno()
        args = [sys.executable, __file__, self.ffmpeg, str(self.fps), str(fd)]
        if self.initial_jpeg:
            handle, self.jpeg_path = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(handle, "wb") as file:
                file.write(self.initial_jpeg)
            args.append(self.jpeg_path)
        self.process = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            pass_fds=(fd,))
        self.tasks = [asyncio.create_task(self._log_stderr())]

    async def _log_stderr(self):
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            self.log(line.decode(errors="replace").rstrip())

    async def feed(self, h264: bytes):
        if self.process and self.process.stdin and not self.process.stdin.is_closing():
            self.process.stdin.write(h264)
            await self.process.stdin.drain()

    async def close(self):
        if self.process:
            if self.process.stdin and not self.process.stdin.is_closing():
                self.process.stdin.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.process.wait(), 3)
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.process:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.process.communicate(), 5)
        if self.jpeg_path:
            with contextlib.suppress(OSError):
                os.unlink(self.jpeg_path)


async def _child(ffmpeg: str, fps: int, fd: int, jpeg: bytes | None):
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=1 << 22)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    # The viewer socket is inherited; write to it directly, never via the plugin.
    _, writer = await asyncio.open_connection(sock=socket.socket(fileno=fd))

    async def output(chunk: bytes):
        writer.write(chunk)
        await writer.drain()

    holder = FrameHolder(ffmpeg, output, fps, jpeg, log=lambda line: print(line, file=sys.stderr, flush=True))
    await holder.start()
    try:
        while chunk := await reader.read(1 << 16):
            await holder.feed(chunk)
    finally:
        await holder.close()


if __name__ == "__main__":
    opening = None
    if len(sys.argv) > 4:
        with open(sys.argv[4], "rb") as file:
            opening = file.read()
    asyncio.run(_child(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), opening))
