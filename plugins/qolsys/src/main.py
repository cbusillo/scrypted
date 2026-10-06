"""Local Qolsys still images and an explicitly enabled bounded video preview."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import socket
from pathlib import Path
from typing import cast

import scrypted_sdk
from qolsys_controller.pairing_server import QolsysPairingServer
from scrypted_sdk.types import (
    Camera,
    FFmpegInput,
    ScryptedInterface,
    Settings,
    VideoCamera,
)

from camera_h264 import H264Parameters
from panel_client import SNAPSHOT_INTERVAL_SECONDS, PanelClient
from preview_server import PreviewServer


class PanelOnlyPairing(QolsysPairingServer):
    async def handle_client(self, reader, writer):
        if writer.get_extra_info("peername")[0] != self._settings.panel_ip:
            writer.close()
            await writer.wait_closed()
            return
        await super().handle_client(reader, writer)


class QolsysCamera(scrypted_sdk.ScryptedDeviceBase, Camera, VideoCamera, Settings):
    def __init__(self, native_id=None):
        super().__init__(native_id)
        self.client = None
        self.session = None
        self._session_watch = None
        self.configuration_lock = asyncio.Lock()

    def _parameters(self):
        value = self.storage.getItem("codecParameters")
        if not value:
            return None
        parsed = json.loads(value)
        return H264Parameters(parsed["lengthSize"], tuple(base64.b64decode(item, validate=True)
                                                          for item in parsed["parameterSets"]),
                              parsed.get("frameRate"))

    def _save_parameters(self, parameters):
        self.storage.setItem("codecParameters", json.dumps({
            "lengthSize": parameters.length_size,
            "parameterSets": [base64.b64encode(item).decode() for item in parameters.parameter_sets],
            "frameRate": parameters.frame_rate,
        }))

    def _snapshot_interval(self):
        try:
            return max(5, int(self.storage.getItem("snapshotInterval") or SNAPSHOT_INTERVAL_SECONDS))
        except ValueError:
            return SNAPSHOT_INTERVAL_SECONDS

    async def _get_client(self):
        if self.client is not None:
            return self.client
        host = self.storage.getItem("panelIp")
        remote = self.storage.getItem("remoteMac")
        if not host or not remote:
            raise RuntimeError("Configure the panel address and pair IQ Remote first")
        local = self.storage.getItem("localIp")
        if not local:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect((host, 8883))
                local = route.getsockname()[0]
        directory = Path(os.environ["SCRYPTED_PLUGIN_VOLUME"]) / "qolsys"
        self.client = PanelClient(directory, host, local, remote)
        self.client.snapshot_interval = self._snapshot_interval()
        return self.client

    async def getPictureOptions(self):
        return [{"id": "panel", "name": "Panel camera", "width": 1280, "height": 720}]

    async def takePicture(self, options=None):
        client = await self._get_client()
        jpeg = await client.take_picture()
        return await scrypted_sdk.mediaManager.createMediaObject(jpeg, "image/jpeg")

    async def getVideoStreamOptions(self):
        parameters = self._parameters()
        if self.storage.getItem("previewEnabled") != "true" or not parameters:
            return []
        return [{"id": "native-preview", "name": "Experimental 12-second live preview",
                 "container": "h264", "video": {"codec": "h264", "width": 1280, "height": 720,
                                                   "fps": round(parameters.frame_rate or 10)}, "audio": None}]

    async def _save_session_parameters(self, session):
        """Keep decoder parameters current; the panel may change them with quality or firmware."""
        await session.finished.wait()
        if session.parameters and session.parameters != self._parameters():
            self._save_parameters(session.parameters)

    async def getVideoStream(self, options=None):
        streams = await self.getVideoStreamOptions()
        if not streams:
            raise RuntimeError("Enable experimental preview and calibrate video first")
        if options and options.get("id") not in (None, "native-preview"):
            raise ValueError("Unknown video stream")
        if self.session and not self.session.finished.is_set():
            raise RuntimeError("Only one live preview viewer is supported")
        self.session = PreviewServer(await self._get_client(), self._parameters())
        url = await self.session.start()
        self._session_watch = asyncio.create_task(self._save_session_parameters(self.session))
        # The generated Python TypedDict makes TypeScript's optional FFmpeg
        # keys required. This payload follows the actual optional-field API.
        ffmpeg_input = cast(FFmpegInput, {
            "url": url, "container": "h264",
            "inputArguments": ["-fflags", "+genpts", "-probesize", "32768", "-analyzeduration", "0",
                               "-f", "h264", "-framerate", str(streams[0]["video"]["fps"]), "-i", url],
            "mediaStreamOptions": streams[0],
        })
        return await scrypted_sdk.mediaManager.createFFmpegMediaObject(ffmpeg_input)

    async def getSettings(self):
        return [
            {"key": "previewStatus", "title": "Preview status", "readonly": True,
             "value": self.session.state if self.session else "Ready"},
            {"key": "panelIp", "title": "Panel IP address", "value": self.storage.getItem("panelIp") or ""},
            {"key": "localIp", "title": "Local LAN address", "description": "Leave empty to select the route to the panel.",
             "value": self.storage.getItem("localIp") or ""},
            {"key": "remoteMac", "title": "IQ Remote identity", "readonly": True,
             "value": self.storage.getItem("remoteMac") or "Not paired"},
            {"key": "pair", "title": "Pair IQ Remote", "type": "button",
             "description": "Start this listener, then press Pair under IQ Remote Devices on the panel."},
            {"key": "snapshotInterval", "title": "Minimum seconds between panel snapshots", "type": "number",
             "value": self._snapshot_interval(),
             "description": "Each snapshot writes and deletes a file on the panel. Viewers share the latest image."},
            {"key": "previewEnabled", "title": "Experimental live preview", "type": "boolean",
             "value": self.storage.getItem("previewEnabled") == "true",
             "description": "One 12-second view while disarmed on mains power. Pauses panel camera motion only if it was on; downloads can lag. Continuous recording is not supported."},
            {"key": "calibrate", "title": "Calibrate video", "type": "button",
             "description": "Record and remove one eight-second clip to prepare the live decoder. Panel must be disarmed."},
        ]

    async def putSetting(self, key, value):
        async with self.configuration_lock:
            if self.session and not self.session.finished.is_set():
                raise RuntimeError("Wait for the current preview to finish before changing settings")
            if key in {"panelIp", "localIp"}:
                if self.client:
                    await self.client.close()
                self.client = None
                self.storage.setItem(key, str(value).strip())
                self.storage.removeItem("codecParameters")
            elif key == "snapshotInterval":
                self.storage.setItem(key, str(max(5, int(value))))
                if self.client:
                    self.client.snapshot_interval = self._snapshot_interval()
            elif key == "previewEnabled":
                self.storage.setItem(key, "true" if value is True else "false")
            elif key == "pair":
                await self._pair()
            elif key == "calibrate":
                if self.storage.getItem("previewEnabled") != "true":
                    raise RuntimeError("Enable experimental preview before video calibration")
                parameters = await (await self._get_client()).capture_preview(8)
                self._save_parameters(parameters)
            else:
                raise ValueError("Unknown setting")
        await self.onDeviceEvent(ScryptedInterface.Settings, None)

    async def _pair(self):
        if not self.storage.getItem("panelIp"):
            raise RuntimeError("Configure the panel address first")
        if not self.storage.getItem("remoteMac"):
            remote = bytes([2]) + secrets.token_bytes(5)
            self.storage.setItem("remoteMac", ":".join(f"{byte:02X}" for byte in remote))
        client = await self._get_client()
        controller = client.controller
        pki = controller.pki
        if not pki.key_file_path.exists():
            if pki.cer_file_path.exists() or pki.secure_file_path.exists():
                raise RuntimeError("Incomplete pairing files preserved; restore the existing private key")
            if not await pki.create(controller.settings.random_mac, key_size=2048):
                raise RuntimeError("Pairing certificate creation failed")
        controller.settings.pairing_timeout = 180
        server = PanelOnlyPairing(controller.settings, pki)
        if not (pki.secure_file_path.exists() and pki.qolsys_cer_file_path.exists()):
            try:
                await server.start()
                self.print("Pairing listener ready. Press Pair under IQ Remote Devices on the panel.")
                await server.wait_until_paired()
            finally:
                await server.stop()
        await client.connect()
        result = await controller.commands.panel.connect()
        if result.get("responseStatus") not in (True, "true"):
            raise RuntimeError("IQ Remote registration was not acknowledged")
        self.print("IQ Remote pairing complete.")


def create_scrypted_plugin():
    return QolsysCamera()
