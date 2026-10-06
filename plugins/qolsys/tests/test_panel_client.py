"""Exercise camera ownership, bounded cleanup and cancellation using fake IO."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from camera_h264 import H264Parameters
from panel_client import DESCRIPTION, SETTINGS_URI, STATE_URI, PanelClient


def box(kind, payload):
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def video():
    sps, pps = b"\x67\x42", b"\x68\x01"
    config = b"\x01\x42\x00\x1e\xff\xe1\x00\x02" + sps + b"\x01\x00\x02" + pps
    entry = box(b"avc1", bytes(78) + box(b"avcC", config))
    moov = box(b"moov", box(b"stsd", bytes(4) + b"\x00\x00\x00\x01" + entry))
    return box(b"ftyp", b"isom") + moov + box(b"mdat", b"\x00\x00\x00\x04\x65abc")


OWN_ID = "6f1c1d6e-2b9a-4a35-9d6b-7c2b0f5d1a10"


def transactions(client):
    return [fields["ipcTransactionID"] for event, fields in client.calls if event == "ipcCall"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    c = PanelClient(tmp_path, "192.0.2.1", "192.0.2.2", "02:00:00:00:00:01")
    c.connect = AsyncMock()
    c.controller.commands.camera.capture_snapshot = AsyncMock(
        return_value=SimpleNamespace(jpeg=b"fresh jpeg", filename="clock_100.jpg"))
    c.calls = []
    c.ident = None
    c.removed = False
    c.deleted = False
    c.armed = False
    c.alarm = "None"
    c.ac = "ON"
    c.motion = "true"
    c.stopped = False

    async def request(event, fields):
        c.calls.append((event, fields))
        if event == "ipcCall":
            if fields["ipcTransactionID"] == 1:
                c.ident = fields["ipcRequest"][1]["dataValue"]
            if fields["ipcTransactionID"] == 2:
                c.stopped = True  # Native stop gives the row its callback filename.
            if fields["ipcTransactionID"] == 7:
                c.removed = True
            return {"responseStatus": "success"}
        if event == "database" and fields["uri"] == SETTINGS_URI:
            values = {"SYSTEM_STATUS": "ARM_AWAY" if c.armed else "DISARM", "PANEL_IP_CAMERA_STATUS": "false",
                      "PANEL_MOTION_DETECTOR": c.motion, "AC_STATUS": c.ac}
            return {"responseStatus": "success", "resultSet": [
                {"name": name, "value": value} for name, value in values.items() if name in fields["selection"]]}
        if event == "database" and fields["uri"] == STATE_URI:
            return {"responseStatus": "success", "resultSet": [{"value": c.alarm}]}
        if event == "database" and fields["dbOperation"] == "insert":
            return {"longValue": 1}
        if event == "database" and fields["dbOperation"] == "delete":
            c.deleted = True
            return {"responseStatus": "success"}
        if event == "database" and fields["dbOperation"] == "read":
            if c.deleted:
                return {"responseStatus": "success", "resultSet": []}
            return {"responseStatus": "success", "resultSet": [{"name": c.ident + "_100.mp4" if c.stopped else ""}]}
        if event == "photoFrameImageDownloadRequest":
            response = {"eventName": event, "directory": fields["directory"], "photoFrameImageName": ""}
            if not c.removed and fields["photoFrameImageName"].endswith("_100.mp4"):
                import base64
                response["photoFrameImageString"] = base64.b64encode(video()).decode()
            return response
        raise AssertionError("Unexpected IO")

    c.request = request
    guard = SimpleNamespace(returncode=None)
    guard.terminate = lambda: setattr(guard, "returncode", 0)
    guard.wait = AsyncMock(return_value=0)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=guard))
    c.guard = guard
    return c


@pytest.mark.asyncio
async def test_still_requests_share_cache(client):
    assert await client.take_picture() == b"fresh jpeg"
    assert await client.take_picture() == b"fresh jpeg"
    client.controller.commands.camera.capture_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_busy_camera_does_not_capture(client):
    async with client.camera_lock:
        with pytest.raises(RuntimeError, match="using the panel camera"):
            await client.take_picture()
    client.controller.commands.camera.capture_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_armed_panel_never_starts(client):
    client.armed = True
    with pytest.raises(RuntimeError, match="disarmed"):
        await client.capture_preview(4)
    assert all(event != "ipcCall" for event, _ in client.calls)


@pytest.mark.asyncio
async def test_emit_failure_stops_resumes_and_removes_only_own_file(client):
    async def emit(packet):
        assert packet.startswith(b"\x00\x00\x00\x01")
        raise ConnectionError("Viewer disconnected")
    with pytest.raises(ConnectionError):
        await client.capture_preview(4, emit, H264Parameters.from_mp4(video()))
    assert transactions(client) == [1, 2, 4, 7]
    assert client.removed and client.deleted and client.guard.returncode == 0
    delete = next(fields for event, fields in client.calls if event == "database" and fields["dbOperation"] == "delete")
    assert client.ident in delete["selection"] and "user_id=-2" in delete["selection"]
    assert DESCRIPTION in delete["selection"]


@pytest.mark.asyncio
async def test_cancellation_finishes_file_cleanup(client):
    emitting = asyncio.Event()
    async def emit(packet):
        emitting.set()
        await asyncio.Future()
    task = asyncio.create_task(client.capture_preview(4, emit, H264Parameters.from_mp4(video())))
    await asyncio.wait_for(emitting.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.removed and client.deleted and client.guard.returncode == 0


@pytest.mark.asyncio
async def test_stop_failure_still_attempts_motion_resume(client):
    calls = []
    async def ipc(transaction, args):
        calls.append(transaction)
        if transaction == 2:
            raise ConnectionError("Stop lost")
    client.ipc = ipc
    with pytest.raises(RuntimeError, match="needs recovery"):
        await client.stop_recording(OWN_ID, resume_motion=True)
    assert calls == [2, 4]


@pytest.mark.asyncio
async def test_cleanup_rejects_traversal_before_ipc(client):
    with pytest.raises(ValueError):
        await client.cleanup_video("../outside", "../outside_100.mp4")
    assert client.calls == []


@pytest.mark.asyncio
async def test_calibration_ends_at_bound_and_removes_clip(client):
    parameters = await client.capture_preview(4)
    assert parameters == H264Parameters.from_mp4(video())
    assert client.removed and client.deleted


@pytest.mark.asyncio
async def test_arming_aborts_and_stops_own_recording(client):
    async def emit(packet):
        client.armed = True
    with pytest.raises(RuntimeError, match="disarmed"):
        await client.capture_preview(5, emit, H264Parameters.from_mp4(video()))
    assert transactions(client) == [1, 2, 4, 7]
    assert client.removed and client.deleted


@pytest.mark.asyncio
async def test_alarm_never_stops_the_panel_camera(client):
    async def emit(packet):
        client.alarm = "Alarm"
    with pytest.raises(RuntimeError):
        await client.capture_preview(5, emit, H264Parameters.from_mp4(video()))
    assert 2 not in transactions(client) and 4 not in transactions(client)
    assert client.guard.returncode is None  # Watchdog keeps retrying after the alarm.


@pytest.mark.asyncio
async def test_preempted_recording_is_not_stopped_but_own_file_is_removed(client):
    async def emit(packet):
        client.stopped = True  # Panel's own recording ended ours.
    await client.capture_preview(4, emit, H264Parameters.from_mp4(video()))
    assert transactions(client) == [1, 7]
    assert client.removed and client.deleted and client.guard.returncode == 0


@pytest.mark.asyncio
async def test_disabled_camera_motion_is_not_turned_on(client):
    client.motion = "false"
    await client.capture_preview(4)
    assert transactions(client) == [1, 2, 7]


@pytest.mark.asyncio
async def test_motion_is_not_resumed_after_power_loss(client):
    async def emit(packet):
        client.ac = "OFF"
    with pytest.raises(RuntimeError, match="battery"):
        await client.capture_preview(4, emit, H264Parameters.from_mp4(video()))
    assert transactions(client) == [1, 2, 7]


@pytest.mark.asyncio
async def test_battery_power_never_starts(client):
    client.ac = "OFF"
    with pytest.raises(RuntimeError, match="battery"):
        await client.capture_preview(4)
    assert transactions(client) == []


@pytest.mark.asyncio
async def test_preview_longer_than_request_bound_is_rejected(client):
    with pytest.raises(ValueError):
        await client.capture_preview(20)
    assert client.calls == []


@pytest.mark.asyncio
async def test_snapshot_cache_follows_interval(client, monkeypatch):
    client.snapshot_interval = 30
    now = [1000.0]
    monkeypatch.setattr("panel_client.time.monotonic", lambda: now[0])
    await client.take_picture()
    now[0] += 29
    await client.take_picture()
    assert client.controller.commands.camera.capture_snapshot.await_count == 1
    now[0] += 2
    await client.take_picture()
    assert client.controller.commands.camera.capture_snapshot.await_count == 2


@pytest.mark.asyncio
async def test_absence_reply_must_match_request(client):
    client.request = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="absence response"):
        await client.read_video("own.mp4")


@pytest.mark.asyncio
async def test_watchdog_spawn_failure_removes_record_without_starting_camera(client, monkeypatch):
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(side_effect=OSError("spawn failed")))
    with pytest.raises(OSError, match="spawn failed"):
        await client.capture_preview(4)
    assert client.deleted
    assert all(event != "ipcCall" for event, _ in client.calls)
    delete = next(fields for event, fields in client.calls if event == "database" and fields["dbOperation"] == "delete")
    assert "user_id=-2" in delete["selection"] and DESCRIPTION in delete["selection"]


@pytest.mark.asyncio
async def test_cleanup_video_tolerates_eventually_consistent_readback(client, monkeypatch):
    monkeypatch.setattr("panel_client.asyncio.sleep", AsyncMock())
    ident = OWN_ID
    name = f"{ident}_100.mp4"
    reads = {"video": 0, "row": 0}
    ipc_calls = []

    async def ipc(transaction, args):
        ipc_calls.append(transaction)

    async def read_video(filename):
        reads["video"] += 1
        return b"stillhere" if reads["video"] == 1 else None  # stale once, then gone

    async def request(event, fields):
        if event == "database" and fields["dbOperation"] == "delete":
            return {"responseStatus": "success"}
        if event == "database" and fields["dbOperation"] == "read":
            reads["row"] += 1
            return {"responseStatus": "success", "resultSet": [{"name": name}] if reads["row"] == 1 else []}
        raise AssertionError(event)

    client.ipc = ipc
    client.read_video = read_video
    client.request = request
    await client.cleanup_video(ident, name)  # must not raise despite first stale read
    assert ipc_calls == [7] and reads["video"] > 1 and reads["row"] > 1
