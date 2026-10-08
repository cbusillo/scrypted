"""Turn bursty panel H264 into a steady stream that viewers can pace.

The panel only offers whole-file downloads of a growing recording, so frames
arrive in clumps with silent gaps. Viewers such as Apple Home read silence as
a failing network. FrameHolder decodes the bursty input, keeps the newest
picture, and re-encodes exactly `fps` pictures per second, repeating the last
picture through gaps.

The plugin runs it in its own process (HolderProcess) so panel downloads in
the plugin's event loop cannot starve the steady clock. That process owns the
listening socket and sends the same stream to every viewer; Apple Home often
opens two streams for one camera.
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
# The child ends once no viewer has joined this long after start, or the last
# viewer has been gone this long; Home reconnects within a second or two.
FIRST_VIEWER_SECONDS = 15
LAST_VIEWER_GRACE_SECONDS = 3
# Drop a viewer that falls this far behind instead of buffering for it.
MAX_VIEWER_BACKLOG = 4_000_000
SPS = 7
BITRATE_KBPS = 1000
KEYFRAME_SECONDS = 2


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
            # Home asks for a few hundred kbit/s and stalls on multi-megabit keyframe
            # bursts, so cap the rate and send a keyframe every two seconds. Repeat
            # SPS/PPS before every keyframe so a viewer can join mid-stream.
            "-b:v", f"{BITRATE_KBPS}k", "-maxrate", f"{BITRATE_KBPS}k", "-bufsize", f"{BITRATE_KBPS}k",
            "-g", str(self.fps * KEYFRAME_SECONDS), "-bf", "0", "-x264-params", "repeat-headers=1",
            "-flush_packets", "1", "-f", "h264", "pipe:1",
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
    """Run FrameHolder in a child process that accepts viewers on a listening socket."""

    def __init__(self, ffmpeg: str, listener: socket.socket, fps: int = 10,
                 initial_jpeg: bytes | None = None, log: Callable[[str], None] = print):
        self.ffmpeg = ffmpeg
        self.listener = listener
        self.fps = fps
        self.initial_jpeg = initial_jpeg
        self.log = log
        self.process: asyncio.subprocess.Process | None = None
        self.jpeg_path: str | None = None
        self.tasks: list[asyncio.Task] = []
        # Set when the child exits and its stderr closes. Process.wait() would
        # also wait for our open stdin pipe.
        self.exited = asyncio.Event()

    async def start(self):
        fd = self.listener.fileno()
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
        try:
            while line := await self.process.stderr.readline():
                self.log(line.decode(errors="replace").rstrip())
        finally:
            self.exited.set()

    async def feed(self, h264: bytes):
        if self.exited.is_set():
            raise ConnectionError("Every viewer has left")
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


class Broadcast:
    """Send whole H264 NAL units to every viewer; a new viewer starts at the next SPS."""

    def __init__(self):
        self.viewers: dict[asyncio.StreamWriter, bool] = {}  # writer -> waiting for a keyframe
        self.pending = bytearray()

    def add(self, writer: asyncio.StreamWriter):
        self.viewers[writer] = True

    def remove(self, writer: asyncio.StreamWriter):
        self.viewers.pop(writer, None)

    def feed(self, data: bytes):
        self.pending.extend(data)
        starts = []
        index = self.pending.find(b"\x00\x00\x01")
        while index != -1:
            starts.append(index)
            index = self.pending.find(b"\x00\x00\x01", index + 3)
        # The last unit is complete only once the next start code arrives.
        for begin, end in zip(starts, starts[1:]):
            self._send(bytes(self.pending[begin + 3:end]).rstrip(b"\x00"))
        if starts:
            del self.pending[:starts[-1]]

    def _send(self, unit: bytes):
        if not unit:
            return
        packet = b"\x00\x00\x00\x01" + unit
        for writer, waiting in list(self.viewers.items()):
            if waiting and unit[0] & 0x1F != SPS:
                continue
            if writer.is_closing() or writer.transport.get_write_buffer_size() > MAX_VIEWER_BACKLOG:
                self.remove(writer)
                writer.close()
                continue
            self.viewers[writer] = False
            writer.write(packet)


async def _child(ffmpeg: str, fps: int, fd: int, jpeg: bytes | None):
    loop = asyncio.get_running_loop()

    def log(line: str):
        print(line, file=sys.stderr, flush=True)

    reader = asyncio.StreamReader(limit=1 << 22)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    broadcast = Broadcast()
    changed = asyncio.Event()

    async def viewer(view_reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        broadcast.add(writer)
        log(f"viewer joined; {len(broadcast.viewers)} watching")
        changed.set()
        with contextlib.suppress(Exception):
            await view_reader.read()  # Viewers never send; this returns when one leaves.
        broadcast.remove(writer)
        writer.close()
        log(f"viewer left; {len(broadcast.viewers)} watching")
        changed.set()

    # The listening socket is inherited; viewers connect here, never via the plugin.
    server = await asyncio.start_server(viewer, sock=socket.socket(fileno=fd))

    async def output(chunk: bytes):
        broadcast.feed(chunk)

    async def watch_viewers():
        joined = False
        while True:
            changed.clear()
            if broadcast.viewers:
                joined = True
                await changed.wait()
                continue
            try:
                await asyncio.wait_for(changed.wait(), LAST_VIEWER_GRACE_SECONDS if joined else FIRST_VIEWER_SECONDS)
            except TimeoutError:
                log("no viewers; ending preview")
                return

    async def feed():
        while chunk := await reader.read(1 << 16):
            await holder.feed(chunk)

    holder = FrameHolder(ffmpeg, output, fps, jpeg, log=log)
    await holder.start()
    tasks = [asyncio.create_task(watch_viewers()), asyncio.create_task(feed())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        server.close()
        for writer in list(broadcast.viewers):
            writer.close()
        await holder.close()


if __name__ == "__main__":
    opening = None
    if len(sys.argv) > 4:
        with open(sys.argv[4], "rb") as file:
            opening = file.read()
    asyncio.run(_child(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), opening))
