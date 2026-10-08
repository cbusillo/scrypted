"""Exercise Scrypted-facing settings and media objects using its real SDK types."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import scrypted_sdk

from camera_h264 import H264Parameters
from main import QolsysCamera


class Storage:
    def __init__(self):
        self.items = {}

    def getItem(self, key):
        return self.items.get(key)

    def setItem(self, key, value):
        self.items[key] = value

    def removeItem(self, key):
        self.items.pop(key, None)


@pytest.fixture
def camera(monkeypatch):
    storage = Storage()
    monkeypatch.setattr(scrypted_sdk, "deviceManager", SimpleNamespace(
        getDeviceStorage=lambda _: storage, onDeviceEvent=AsyncMock()))
    monkeypatch.setattr(scrypted_sdk, "mediaManager", SimpleNamespace(
        createMediaObject=AsyncMock(side_effect=lambda data, mime: (data, mime)),
        createFFmpegMediaObject=AsyncMock(side_effect=lambda data: data)))
    return QolsysCamera()


@pytest.mark.asyncio
async def test_snapshot_uses_fresh_controller_bytes(camera):
    camera.client = SimpleNamespace(take_picture=AsyncMock(return_value=b"new JPEG"))
    assert await camera.takePicture() == (b"new JPEG", "image/jpeg")
    camera.client.take_picture.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_is_not_advertised_without_explicit_enable_and_calibration(camera):
    assert await camera.getVideoStreamOptions() == []
    camera.storage.setItem("previewEnabled", "true")
    assert await camera.getVideoStreamOptions() == []
    with pytest.raises(RuntimeError, match="calibrate"):
        await camera.getVideoStream()


@pytest.mark.asyncio
async def test_calibration_is_opt_in_and_saves_only_parameters(camera):
    parameters = H264Parameters(4, (b"\x67sps", b"\x68pps"))
    camera.client = SimpleNamespace(capture_preview=AsyncMock(return_value=parameters))
    with pytest.raises(RuntimeError, match="Enable experimental"):
        await camera.putSetting("calibrate", True)
    camera.client.capture_preview.assert_not_awaited()
    await camera.putSetting("previewEnabled", True)
    await camera.putSetting("calibrate", True)
    assert camera._parameters() == parameters
    assert isinstance(camera.storage.getItem("codecParameters"), str)
    assert await camera.getVideoStreamOptions()


@pytest.mark.asyncio
async def test_unknown_stream_does_not_open_listener(camera):
    await camera.putSetting("previewEnabled", True)
    camera._save_parameters(H264Parameters(4, (b"\x67sps", b"\x68pps")))
    with pytest.raises(ValueError, match="Unknown video"):
        await camera.getVideoStream({"id": "other"})
    assert camera.session is None


@pytest.mark.asyncio
async def test_preview_refreshes_stored_decoder_parameters(camera):
    old = H264Parameters(4, (b"\x67old", b"\x68old"), 10.0)
    new = H264Parameters(4, (b"\x67new", b"\x68new"), 15.0)
    camera._save_parameters(old)
    session = SimpleNamespace(finished=asyncio.Event(), parameters=new)
    session.finished.set()
    await camera._save_session_parameters(session)
    assert camera._parameters() == new


@pytest.mark.asyncio
async def test_stream_uses_recorded_frame_rate(camera):
    await camera.putSetting("previewEnabled", True)
    camera._save_parameters(H264Parameters(4, (b"\x67sps", b"\x68pps"), 15.0))
    assert (await camera.getVideoStreamOptions())[0]["video"]["fps"] == 15


@pytest.mark.asyncio
async def test_new_viewer_request_takes_over_the_running_preview(camera, monkeypatch):
    import main
    await camera.putSetting("previewEnabled", True)
    camera._save_parameters(H264Parameters(4, (b"\x67sps", b"\x68pps"), 10.0))
    camera.client = SimpleNamespace(last_jpeg=None)
    camera.print = lambda *args: None
    scrypted_sdk.mediaManager.getFFmpegPath = AsyncMock(return_value="ffmpeg")
    sessions = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.finished = asyncio.Event()
            self.parameters = None
            sessions.append(self)

        async def start(self):
            return "tcp://127.0.0.1:1"

        async def close(self):
            self.finished.set()

    monkeypatch.setattr(main, "PreviewServer", Session)
    await camera.getVideoStream()
    await camera.getVideoStream()
    assert len(sessions) == 2
    assert sessions[0].finished.is_set() and not sessions[1].finished.is_set()
    assert camera.session is sessions[1]
