"""A single local consumer for newly recorded H264 preview segments.

Each segment is one bounded native recording. With chaining enabled, the next
segment starts as soon as one ends, for as long as the viewer stays connected
up to max_view_seconds. With ffmpeg available, a FrameHolder process turns the bursty
segments into a steady stream that repeats the last picture through gaps.
"""

from __future__ import annotations

import asyncio
import contextlib

from frame_holder import HolderProcess

DRAIN_TIMEOUT = 5
CLOSE_TIMEOUT = 2


class PreviewServer:
    def __init__(self, client, parameters, seconds=12, max_view_seconds=0, fps=10, ffmpeg=None, initial_jpeg=None,
                 log=print):
        self.client = client
        self.parameters = parameters
        self.seconds = seconds
        self.max_view_seconds = max_view_seconds
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.initial_jpeg = initial_jpeg
        self.log = log
        self.server = None
        self.task = None
        self.started = False
        self.finished = asyncio.Event()
        self.error = None
        self.started_at = 0.0
        self.state = "Waiting for viewer"

    async def start(self):
        self.started_at = asyncio.get_running_loop().time()
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.task = asyncio.create_task(self._expire_unclaimed())
        return f"tcp://127.0.0.1:{port}"

    async def _expire_unclaimed(self):
        await asyncio.sleep(15)
        if not self.started:
            await self.close()

    async def _handle(self, reader, writer):
        if self.started:
            writer.close()
            await writer.wait_closed()
            return
        self.started = True
        self.state = "Capturing preview"
        self.server.close()
        if self.task:
            self.task.cancel()
        self.task = asyncio.current_task()

        async def write(packet):
            writer.write(packet)
            async with asyncio.timeout(DRAIN_TIMEOUT):
                await writer.drain()

        holder = (HolderProcess(self.ffmpeg, writer.get_extra_info("socket"), self.fps, self.initial_jpeg, self.log)
                  if self.ffmpeg else None)

        async def emit(packet):
            if holder:
                await holder.feed(packet)
            else:
                await write(packet)

        async def produce():
            if holder:
                await holder.start()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.max_view_seconds
            try:
                while True:
                    self.parameters = await self.client.capture_preview(self.seconds, emit, self.parameters)
                    if loop.time() >= deadline:
                        return self.parameters
            finally:
                if holder:
                    await holder.close()

        producer = asyncio.create_task(produce())
        disconnected = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait({producer, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done and not producer.done():
                self.log("viewer disconnected; stopping preview")
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                self.parameters = await producer
        except Exception as error:
            self.error = error
            self.log(f"preview ended with error: {error!r}")
        finally:
            self.state = "Finishing preview"
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer
            self.state = "Closing viewer"
            disconnected.cancel()
            # Close the socket before waiting for its reader: a transport can
            # defer read cancellation until the connection itself is closed.
            writer.close()
            try:
                async with asyncio.timeout(CLOSE_TIMEOUT):
                    await writer.wait_closed()
            except (ConnectionError, TimeoutError):
                writer.transport.abort()
            finally:
                self.state = "Ready" if self.error is None else "Preview failed"
                self.finished.set()

    async def close(self):
        if self.server:
            self.server.close()
        if self.task and self.task is not asyncio.current_task():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        if self.server:
            await self.server.wait_closed()
        if not self.started:
            self.state = "Ready"
            self.finished.set()
