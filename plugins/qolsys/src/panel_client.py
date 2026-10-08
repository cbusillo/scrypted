"""Camera-only IQ Remote transport and bounded native preview experiments."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from qolsys_controller.commands.camera import PhotoDirectory
from qolsys_controller.controller import QolsysController
from qolsys_controller.mqtt_command import MQTTCommand

from camera_h264 import GrowingH264Reader, H264Parameters

CAMERA_URI = "content://com.qolsys.qolsysprovider.CameraRequestContentProvider/camerarequest"
SETTINGS_URI = "content://com.qolsys.qolsysprovider.QolsysSettingsProvider/qolsyssettings"
STATE_URI = "content://com.qolsys.qolsysprovider.StateContentProvider/state"
DESCRIPTION = "Local Scrypted preview"
MAX_ENCODED_BYTES = 20_000_000
# Whole-file downloads run about 0.6 MB/s, so longer clips outgrow the 8-second
# request timeout. Calibration and the viewer preview both stay inside this.
MAX_PREVIEW_SECONDS = 12
SNAPSHOT_INTERVAL_SECONDS = 30
# Each poll downloads the whole growing clip; poll again almost at once, and
# recheck the arming and power state on a slower clock.
POLL_PAUSE_SECONDS = 0.2
PREFLIGHT_SECONDS = 2
# Reuse the panel-minus-local clock offset learned from earlier files instead of
# taking a snapshot before every chained preview segment.
CLOCK_REUSE_SECONDS = 600
# Bound for the sign-in and ping replies, like the other panel requests.
SIGN_IN_TIMEOUT_SECONDS = 8


class PanelClient:
    """Use the common SDK without syncing users, automations or alarm controls."""

    def __init__(self, config_directory: Path, panel_ip: str, plugin_ip: str, remote_mac: str):
        if not re.fullmatch(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}", remote_mac):
            raise ValueError("A paired IQ Remote MAC address is required")
        self.controller = QolsysController()
        settings = self.controller.settings
        settings.config_directory = str(config_directory)
        settings.panel_ip = panel_ip
        settings.plugin_ip = plugin_ip
        settings.random_mac = remote_mac
        settings.mqtt_remote_client_id = "qolsys-scrypted-" + uuid.uuid4().hex[:12]
        self.controller.pki.set_id(remote_mac)
        self.transport = None
        self.tasks: list[asyncio.Task] = []
        self.connection_lock = asyncio.Lock()
        self.log: Callable[[str], None] = print
        self.cleanups: set[asyncio.Future] = set()
        self.camera_lock = asyncio.Lock()
        self.last_jpeg: bytes | None = None
        self.last_picture_time = 0.0
        self.snapshot_interval = SNAPSHOT_INTERVAL_SECONDS
        self.clock_offset: float | None = None
        self.clock_learned = 0.0
        self.recording_started = 0.0

    async def connect(self):
        async with self.connection_lock:
            if self.transport is not None and all(not task.done() for task in self.tasks):
                return
            await self.close()
            client = await self.controller.mqtt_open_transport_task()
            await client.__aenter__()
            try:
                await client.subscribe("response_" + self.controller.settings.random_mac)
            except BaseException:
                await client.__aexit__(None, None, None)
                raise
            self.transport = client

            async def listen():
                async for message in client.messages:
                    response = json.loads(message.payload)
                    if response.get("requestID"):
                        await self.controller.mqtt_command_queue.handle_response(response)

            self.tasks = [asyncio.create_task(self.controller.mqtt_publish_task(client)),
                          asyncio.create_task(listen())]
            # Sign in and ping like the SDK's own session, without its full database
            # sync. The panel lists a remote as Active only while it pings. This is
            # only for the panel's device list, so camera requests never wait on it.
            self.tasks.append(asyncio.create_task(self._keep_active()))

    async def _sign_in(self):
        result = await self.controller.commands.panel.connect()
        self.log(f"Panel sign-in: responseStatus={result.get('responseStatus')!r}, fields={sorted(result)}")

    async def _keep_active(self):
        try:
            async with asyncio.timeout(SIGN_IN_TIMEOUT_SECONDS):
                await self._sign_in()
        except Exception as err:
            self.log(f"Panel sign-in failed: {err!r}")
        while True:
            try:
                async with asyncio.timeout(SIGN_IN_TIMEOUT_SECONDS):
                    await self.controller.commands.panel.pingevent()
            except Exception as err:
                self.log(f"Panel ping failed: {err!r}")
            await asyncio.sleep(self.controller.settings.mqtt_ping)
            # Stop with the connection; the next request reconnects and signs in again.
            if any(task.done() for task in self.tasks if task is not asyncio.current_task()):
                return

    async def close(self):
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.tasks = []
        if self.transport is not None:
            with contextlib.suppress(Exception):
                await self.transport.__aexit__(None, None, None)
            self.transport = None

    async def request(self, event: str, fields: dict):
        await self.connect()
        command = MQTTCommand(self.controller, event)
        for key, value in fields.items():
            command.append(key, value)
        async with asyncio.timeout(8):
            return await command.send_command()

    async def ipc(self, transaction: int, arguments: list[dict]):
        result = await self.request("ipcCall", {
            "ipcServiceName": "qcamservice", "ipcInterfaceName": "qcamservice",
            "ipcTransactionID": transaction, "ipcRequest": arguments,
        })
        if result.get("responseStatus") != "success":
            raise RuntimeError(f"Camera transaction {transaction} was not acknowledged")

    def _cached_picture(self) -> bytes | None:
        if self.last_jpeg and time.monotonic() - self.last_picture_time < self.snapshot_interval:
            return self.last_jpeg
        return None

    async def take_picture(self) -> bytes:
        # Each capture writes and deletes a panel file; serve recent images.
        if cached := self._cached_picture():
            return cached
        if self.camera_lock.locked():
            # A live view holds the camera; a slightly older picture beats a gray tile.
            if self.last_jpeg:
                return self.last_jpeg
            raise RuntimeError("Live preview is using the panel camera; retry after it ends")
        async with self.camera_lock:
            # Recheck after waiting for another picture request.
            if cached := self._cached_picture():
                return cached
            await self.connect()
            snapshot = await self.controller.commands.camera.capture_snapshot()
            self.last_jpeg = snapshot.jpeg
            self.last_picture_time = time.monotonic()
            # Thumbnails keep the panel clock fresh so a later preview skips its clock picture.
            self._learn_clock(int(snapshot.filename.removesuffix(".jpg").rsplit("_", 1)[1]), time.time())
            return snapshot.jpeg

    async def preflight(self) -> bool:
        """Require an idle, disarmed, mains-powered panel; return its motion setting.

        The panel stops camera motion detection on battery and only resumes it
        when PANEL_MOTION_DETECTOR is on, so the preview must not override either.
        """
        flags = await self._settings("SYSTEM_STATUS", "PANEL_IP_CAMERA_STATUS", "PANEL_MOTION_DETECTOR", "AC_STATUS")
        if (flags.get("SYSTEM_STATUS") != "DISARM" or flags.get("PANEL_IP_CAMERA_STATUS", "").lower() != "false"
                or await self._alarm_states() != {"None"}):
            raise RuntimeError("Preview requires a disarmed panel with no alarm and continuous recording off")
        if flags.get("AC_STATUS") != "ON":
            raise RuntimeError("Preview is unavailable while the panel is on battery")
        motion = flags.get("PANEL_MOTION_DETECTOR", "").lower()
        if motion not in ("true", "false"):
            raise RuntimeError("Panel camera motion setting is unavailable")
        return motion == "true"

    async def _settings(self, *names: str) -> dict[str, str]:
        quoted = ",".join(f"'{name}'" for name in names)
        response = await self.request("database", {
            "dbOperation": "read", "uri": SETTINGS_URI, "projection": "[name,value]",
            "selection": f"name IN ({quoted})",
        })
        return {row["name"]: str(row["value"]) for row in response.get("resultSet", [])}

    async def _alarm_states(self) -> set[str]:
        response = await self.request("database", {
            "dbOperation": "read", "uri": STATE_URI, "projection": "[value]", "selection": "name='ALARM_STATE'",
        })
        return {str(row.get("value")) for row in response.get("resultSet", [])}

    async def owns_recording(self, request_id: str) -> bool:
        """Whether the active native recording is still ours and safe to stop.

        Transaction 2 stops whichever recording is active. The panel's own
        recording first stops ours, which gives our row its callback filename.
        During an alarm or entry delay, always leave the camera to the panel.
        """
        if await self._alarm_states() != {"None"}:
            return False
        response = await self.request("database", {
            "dbOperation": "read", "uri": CAMERA_URI, "projection": "[name]",
            "selection": f"request_id='{request_id}'",
        })
        rows = response.get("resultSet")
        if response.get("responseStatus") != "success" or not isinstance(rows, list) or len(rows) != 1:
            return False
        return isinstance(rows[0], dict) and not rows[0].get("name")

    async def recording_finished(self, request_id: str) -> bool:
        """Whether our recording has ended, shown by its callback filename."""
        response = await self.request("database", {
            "dbOperation": "read", "uri": CAMERA_URI, "projection": "[name]",
            "selection": f"request_id='{request_id}'",
        })
        rows = response.get("resultSet")
        return isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict) and bool(rows[0].get("name"))

    async def stop_recording(self, request_id: str, resume_motion: bool) -> bool:
        """Stop our recording and restore motion; returns False if the panel owns the camera."""
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError("Invalid preview request ID")
        if not await self.owns_recording(request_id):
            return False
        errors = []
        try:
            await self.ipc(2, [])
        except Exception as error:
            errors.append(error)
        try:
            # The panel itself keeps camera motion off on battery.
            if resume_motion and (await self._settings("AC_STATUS")).get("AC_STATUS") == "ON":
                await self.ipc(4, [])
        except Exception as error:
            errors.append(error)
        if errors:
            raise RuntimeError("Recording stop or camera-motion resume needs recovery") from errors[0]
        return True

    async def read_video(self, filename: str) -> bytes | None:
        response = await self.request("photoFrameImageDownloadRequest", {
            "directory": PhotoDirectory.PEEK_IN.value, "photoFrameImageName": filename,
        })
        encoded = response.get("photoFrameImageString")
        if not encoded:
            if (response.get("eventName") != "photoFrameImageDownloadRequest"
                    or response.get("directory") != PhotoDirectory.PEEK_IN.value
                    or response.get("photoFrameImageName") not in (filename, "")):
                raise RuntimeError("Video absence response could not be verified")
            return None
        if not isinstance(encoded, str) or len(encoded) > MAX_ENCODED_BYTES:
            raise RuntimeError("Video download exceeds preview bound")
        return base64.b64decode("".join(encoded.split()), validate=True)

    async def _verify(self, condition, attempts=5, delay=0.4) -> bool:
        """Poll an eventually-consistent panel read until confirmed, bounded.

        The panel content provider is not read-after-write consistent, so a
        just-deleted file or row can still read back briefly. Retry before failing.
        """
        for attempt in range(attempts):
            if await condition():
                return True
            if attempt + 1 < attempts:
                await asyncio.sleep(delay)
        return False

    async def cleanup_video(self, request_id: str, filename: str):
        if (str(uuid.UUID(request_id)) != request_id
                or not re.fullmatch(re.escape(request_id) + r"_[0-9]+\.mp4", filename)):
            raise RuntimeError("Unexpected preview filename; cleanup stopped")
        await self.ipc(7, [{"dataType": "int", "dataValue": 2},
                           {"dataType": "string", "dataValue": "../PeekInPhotos/" + filename}])
        if not await self._verify(lambda: self._video_absent(filename)):
            raise RuntimeError("Preview file cleanup unverified; metadata preserved")
        await self.cleanup_metadata(request_id)

    async def _video_absent(self, filename: str) -> bool:
        return await self.read_video(filename) is None

    async def cleanup_metadata(self, request_id: str):
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError("Invalid preview request ID")
        selection = f"request_id='{request_id}' AND user_id=-2 AND file_type='LOCAL_VIDEO' AND description='{DESCRIPTION}'"
        await self.request("database", {"dbOperation": "delete", "uri": CAMERA_URI,
                                        "selection": selection})

        async def row_absent() -> bool:
            response = await self.request("database", {"dbOperation": "read", "uri": CAMERA_URI,
                                                       "projection": "[name]", "selection": selection})
            return response.get("responseStatus") == "success" and response.get("resultSet") == []

        if not await self._verify(row_absent):
            raise RuntimeError("Preview metadata cleanup unverified")

    async def capture_preview(
        self, seconds: float, emit: Callable[[bytes], Awaitable[None]] | None = None,
        parameters: H264Parameters | None = None, defer_cleanup: bool = False,
    ) -> H264Parameters:
        """One bounded recording. No looping, prebuffering, or unattended recording.

        With defer_cleanup, removing the clip's file and record runs in the
        background so a chained preview can start its next clip at once;
        drain_cleanups() waits for it.
        """
        if not 4 <= seconds <= MAX_PREVIEW_SECONDS:
            raise ValueError(f"Preview duration must be between 4 and {MAX_PREVIEW_SECONDS} seconds")
        if emit is not None and parameters is None:
            raise ValueError("Calibrate codec parameters before requesting live preview")
        if self.camera_lock.locked():
            raise RuntimeError("The panel camera is already busy")
        async with self.camera_lock:
            resume_motion = await self.preflight()
            await self.connect()
            if self.clock_offset is None or time.monotonic() - self.clock_learned >= CLOCK_REUSE_SECONDS:
                # The panel clock drifts from ours; learn it from a fresh picture's filename.
                clock_picture = await self.controller.commands.camera.capture_snapshot()
                self.last_jpeg = clock_picture.jpeg
                self.last_picture_time = time.monotonic()
                self._learn_clock(int(clock_picture.filename.removesuffix(".jpg").rsplit("_", 1)[1]), time.time())
            ident = str(uuid.uuid4())
            now = int(time.time() * 1000)
            metadata = {"request_id": ident, "type": "PEEK_IN", "file_type": "LOCAL_VIDEO",
                        "description": DESCRIPTION, "create_time": now, "update_time": now,
                        "partition_id": 0, "user_id": -2, "zone_id": 0, "camera_source": 129, "imageId": 0}
            inserted = await self.request("database", {"dbOperation": "insert", "uri": CAMERA_URI,
                                                       "contentValues": json.dumps(metadata)})
            if str(inserted.get("longValue")) != "1":
                raise RuntimeError("Preview metadata refused; no recording started")
            settings = self.controller.settings
            environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
            try:
                guard = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).with_name("preview_watchdog.py")),
                    str(settings.config_directory), settings.panel_ip, settings.plugin_ip,
                    settings.random_mac, str(seconds + 20), ident, str(resume_motion).lower(), env=environment,
                    start_new_session=True,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
            except BaseException:
                cleanup = asyncio.create_task(self.cleanup_metadata(ident))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                except Exception as error:
                    raise RuntimeError(f"Preview metadata needs recovery; request ID: {ident}") from error
                raise
            filename = None
            reader = GrowingH264Reader(parameters) if parameters else None
            failure: BaseException | None = None
            try:
                self.recording_started = time.time()
                await self.ipc(1, [{"dataType": "int", "dataValue": 0},
                                   {"dataType": "string", "dataValue": ident},
                                   {"dataType": "string", "dataValue": "/sdcard/PeekInPhotos"},
                                   {"dataType": "string", "dataValue": DESCRIPTION}])
                # A poll that runs past the end still finishes: the panel answers
                # requests in order, so abandoning a download would only delay the stop.
                deadline = time.monotonic() + seconds
                checked = time.monotonic()
                while time.monotonic() < deadline:
                    await asyncio.sleep(POLL_PAUSE_SECONDS)
                    if time.monotonic() - checked >= PREFLIGHT_SECONDS:
                        await self.preflight()
                        checked = time.monotonic()
                    raw = None
                    if filename:
                        raw = await self.read_video(filename)
                    else:
                        # Native time of the recording start, from the learned clock offset.
                        expected = int(self.recording_started + self.clock_offset)
                        for stamp in sorted(range(expected - 3, expected + 4), key=lambda v: abs(v - expected)):
                            candidate = f"{ident}_{stamp}.mp4"
                            raw = await self.read_video(candidate)
                            if raw:
                                filename = candidate
                                break
                    if raw and reader and emit:
                        packet = reader.feed(raw)
                        if packet:
                            await emit(packet)
            except BaseException as error:
                failure = error
            finally:
                # Stop, resume motion, and remove the exact file even on cancellation.
                cleanup = asyncio.create_task(
                    self._finish_preview(ident, filename, reader, guard, resume_motion, defer_cleanup))
                try:
                    final_parameters, tail = await asyncio.shield(cleanup)
                except asyncio.CancelledError as error:
                    failure = failure or error
                    final_parameters, tail = await cleanup
                except Exception as error:
                    raise RuntimeError(f"Preview cleanup needs recovery; request ID: {ident}") from error
            if failure is not None:
                raise failure
            if final_parameters is None:
                # Stopped before the panel named its file; nothing to decode.
                if parameters is None:
                    raise RuntimeError("Preview callback filename unavailable")
                final_parameters = parameters
            if tail and emit:
                await emit(tail)
            return final_parameters

    def _learn_clock(self, native_epoch: int, local_time: float):
        self.clock_offset = native_epoch - local_time
        self.clock_learned = time.monotonic()

    async def _finish_preview(self, ident, filename, reader, guard, resume_motion, defer=False):
        stopped = await self.stop_recording(ident, resume_motion)
        # Keep the watchdog while an alarm holds off stopping our recording.
        if guard.returncode is None and (stopped or await self.recording_finished(ident)):
            guard.terminate()
            await guard.wait()
        # The file keeps the name it had while recording; ask only if it was never found.
        for _ in range(0 if filename else 8):
            response = await self.request("database", {
                "dbOperation": "read", "uri": CAMERA_URI, "projection": "[name]",
                "selection": f"request_id='{ident}'",
            })
            rows = response.get("resultSet", [])
            candidate = rows[0].get("name") if rows else None
            if candidate:
                filename = candidate
                break
            await asyncio.sleep(0.25)
        if not isinstance(filename, str) or not re.fullmatch(re.escape(ident) + r"_[0-9]+\.mp4", filename):
            # A recording stopped within a second may never be named. Remove whatever exists.
            await self._remove(self._remove_unnamed(ident), defer)
            return None, b""
        self._learn_clock(int(filename.removesuffix(".mp4").rsplit("_", 1)[1]), self.recording_started)
        if defer and reader is not None:
            # Chained live view: starting the next clip beats re-downloading this
            # whole file for its last second.
            await self._remove(self.cleanup_video(ident, filename), defer)
            return None, b""
        try:
            raw = await self.read_video(filename)
            if raw is None:
                raise RuntimeError("Preview file unavailable")
            parameters = H264Parameters.from_mp4(raw)
            tail = reader.feed(raw) if reader else b""
        finally:
            await self._remove(self.cleanup_video(ident, filename), defer)
        return parameters, tail

    async def _remove_unnamed(self, ident: str):
        if self.clock_offset is not None:
            expected = int(self.recording_started + self.clock_offset)
            for stamp in sorted(range(expected - 3, expected + 4), key=lambda v: abs(v - expected)):
                candidate = f"{ident}_{stamp}.mp4"
                if await self.read_video(candidate):
                    await self.cleanup_video(ident, candidate)
                    return
        await self.cleanup_metadata(ident)

    async def _remove(self, removal: Awaitable[None], defer: bool):
        if not defer:
            await removal
            return
        task = asyncio.ensure_future(removal)
        self.cleanups.add(task)

        def done(finished: asyncio.Future):
            self.cleanups.discard(finished)
            if not finished.cancelled() and finished.exception():
                self.log(f"Preview file removal failed: {finished.exception()!r}")

        task.add_done_callback(done)

    async def drain_cleanups(self, timeout: float = 30):
        """Wait for background clip removal, bounded."""
        if self.cleanups:
            await asyncio.wait(set(self.cleanups), timeout=timeout)
