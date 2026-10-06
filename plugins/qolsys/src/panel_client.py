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
        self.camera_lock = asyncio.Lock()
        self.last_jpeg: bytes | None = None
        self.last_picture_time = 0.0
        self.snapshot_interval = SNAPSHOT_INTERVAL_SECONDS

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
            raise RuntimeError("Live preview is using the panel camera; retry after it ends")
        async with self.camera_lock:
            # Recheck after waiting for another picture request.
            if cached := self._cached_picture():
                return cached
            await self.connect()
            snapshot = await self.controller.commands.camera.capture_snapshot()
            self.last_jpeg = snapshot.jpeg
            self.last_picture_time = time.monotonic()
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
        parameters: H264Parameters | None = None,
    ) -> H264Parameters:
        """One bounded recording. No looping, prebuffering, or unattended recording."""
        if not 4 <= seconds <= MAX_PREVIEW_SECONDS:
            raise ValueError(f"Preview duration must be between 4 and {MAX_PREVIEW_SECONDS} seconds")
        if emit is not None and parameters is None:
            raise ValueError("Calibrate codec parameters before requesting live preview")
        if self.camera_lock.locked():
            raise RuntimeError("The panel camera is already busy")
        async with self.camera_lock:
            resume_motion = await self.preflight()
            await self.connect()
            clock_picture = await self.controller.commands.camera.capture_snapshot()
            self.last_jpeg = clock_picture.jpeg
            self.last_picture_time = time.monotonic()
            native_epoch = int(clock_picture.filename.removesuffix(".jpg").rsplit("_", 1)[1])
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
            budget = asyncio.timeout(seconds)
            try:
                await self.ipc(1, [{"dataType": "int", "dataValue": 0},
                                   {"dataType": "string", "dataValue": ident},
                                   {"dataType": "string", "dataValue": "/sdcard/PeekInPhotos"},
                                   {"dataType": "string", "dataValue": DESCRIPTION}])
                async with budget:
                    while True:
                        await asyncio.sleep(1)
                        await self.preflight()
                        raw = None
                        if filename:
                            raw = await self.read_video(filename)
                        else:
                            for stamp in range(native_epoch - 2, native_epoch + 9):
                                candidate = f"{ident}_{stamp}.mp4"
                                raw = await self.read_video(candidate)
                                if raw:
                                    filename = candidate
                                    break
                        if raw and reader and emit:
                            packet = reader.feed(raw)
                            if packet:
                                await emit(packet)
            except TimeoutError as error:
                if not budget.expired():
                    failure = error
            except BaseException as error:
                failure = error
            finally:
                # Stop, resume motion, and remove the exact file even on cancellation.
                cleanup = asyncio.create_task(self._finish_preview(ident, filename, reader, guard, resume_motion))
                try:
                    final_parameters, tail = await asyncio.shield(cleanup)
                except asyncio.CancelledError as error:
                    failure = failure or error
                    final_parameters, tail = await cleanup
                except Exception as error:
                    raise RuntimeError(f"Preview cleanup needs recovery; request ID: {ident}") from error
            if failure is not None:
                raise failure
            if tail and emit:
                await emit(tail)
            return final_parameters

    async def _finish_preview(self, ident, filename, reader, guard, resume_motion):
        stopped = await self.stop_recording(ident, resume_motion)
        # Keep the watchdog while an alarm holds off stopping our recording.
        if guard.returncode is None and (stopped or await self.recording_finished(ident)):
            guard.terminate()
            await guard.wait()
        for _ in range(8):
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
            raise RuntimeError("Preview callback filename unavailable")
        try:
            raw = await self.read_video(filename)
            if raw is None:
                raise RuntimeError("Preview file unavailable")
            parameters = H264Parameters.from_mp4(raw)
            tail = reader.feed(raw) if reader else b""
        finally:
            await self.cleanup_video(ident, filename)
        return parameters, tail
