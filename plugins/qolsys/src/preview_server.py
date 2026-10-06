"""A single local consumer for one bounded, newly recorded H264 preview."""

from __future__ import annotations

import asyncio
import contextlib

DRAIN_TIMEOUT = 5
CLOSE_TIMEOUT = 2


class PreviewServer:
    def __init__(self, client, parameters, seconds=12):
        self.client = client
        self.parameters = parameters
        self.seconds = seconds
        self.server = None
        self.task = None
        self.started = False
        self.finished = asyncio.Event()
        self.error = None
        self.state = "Waiting for viewer"

    async def start(self):
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

        async def emit(packet):
            writer.write(packet)
            async with asyncio.timeout(DRAIN_TIMEOUT):
                await writer.drain()

        producer = asyncio.create_task(self.client.capture_preview(self.seconds, emit, self.parameters))
        disconnected = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait({producer, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done and not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                self.parameters = await producer
        except Exception as error:
            self.error = error
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
