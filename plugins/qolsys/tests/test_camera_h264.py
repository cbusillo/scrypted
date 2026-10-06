"""Exercise growing-file parsing without storing private camera footage."""

import pytest

from camera_h264 import GrowingH264Reader, H264Parameters


def box(kind: bytes, body: bytes) -> bytes:
    return (len(body) + 8).to_bytes(4, "big") + kind + body


def parameters() -> H264Parameters:
    return H264Parameters(4, (b"\x67SPS", b"\x68PPS"))


def recording(payload: bytes) -> bytes:
    return box(b"ftyp", b"isom") + box(b"free", b"\x00" * 16) + b"\x3f\x3f\x3f\x3fmdat" + payload


def packet(nal: bytes) -> bytes:
    return len(nal).to_bytes(4, "big") + nal


def test_decoder_parameters_are_found_in_nested_sample_description() -> None:
    config = b"\x01\x64\x00\x28\xff\xe1\x00\x04\x67SPS\x01\x00\x04\x68PPS"
    data = box(
        b"moov",
        box(
            b"trak",
            box(
                b"mdia",
                box(
                    b"minf",
                    box(
                        b"stbl",
                        box(b"stsd", b"\x00" * 4 + b"\x00\x00\x00\x01" + box(b"avc1", b"\x00" * 78 + box(b"avcC", config))),
                    ),
                ),
            ),
        ),
    )
    assert H264Parameters.from_mp4(data) == parameters()


def test_partial_nal_waits_for_completion_without_replaying_frames() -> None:
    reader = GrowingH264Reader(parameters())
    first, second = packet(b"\x65first"), packet(b"\x41second")
    prefix = recording(first + second[:5])
    initial = reader.feed(prefix)
    assert initial == b"\x00\x00\x00\x01\x67SPS\x00\x00\x00\x01\x68PPS\x00\x00\x00\x01\x65first"
    assert reader.feed(prefix) == b""
    assert reader.feed(recording(first + second)) == b"\x00\x00\x00\x01\x41second"


def test_no_decoder_configuration_is_emitted_before_complete_frame() -> None:
    reader = GrowingH264Reader(parameters())
    assert reader.feed(recording(packet(b"\x65frame")[:6])) == b""


def test_recording_truncation_requires_a_new_reader() -> None:
    reader = GrowingH264Reader(parameters())
    reader.feed(recording(packet(b"\x65first")))
    with pytest.raises(ValueError, match="truncated"):
        reader.feed(recording(b""))


@pytest.mark.parametrize("nal", [b"\xffinvalid", b"\x00invalid"])
def test_invalid_nal_is_rejected(nal: bytes) -> None:
    reader = GrowingH264Reader(parameters())
    with pytest.raises(ValueError, match="header"):
        reader.feed(recording(packet(nal)))


def test_oversized_nal_is_rejected_without_waiting_for_allocation() -> None:
    reader = GrowingH264Reader(parameters())
    with pytest.raises(ValueError, match="bounded"):
        reader.feed(recording((100_000_000).to_bytes(4, "big")))


def test_missing_decoder_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="configuration"):
        H264Parameters.from_mp4(box(b"ftyp", b"isom"))


def test_frame_rate_comes_from_completed_clip_timing() -> None:
    config = b"\x01\x64\x00\x28\xff\xe1\x00\x04\x67SPS\x01\x00\x04\x68PPS"
    mdhd = box(b"mdhd", bytes(12) + (90000).to_bytes(4, "big") + bytes(8))
    stts = box(b"stts", bytes(4) + (1).to_bytes(4, "big") + (150).to_bytes(4, "big") + (6000).to_bytes(4, "big"))
    entry = box(b"avc1", bytes(78) + box(b"avcC", config))
    stbl = box(b"stbl", box(b"stsd", bytes(4) + (1).to_bytes(4, "big") + entry) + stts)
    data = box(b"moov", box(b"trak", box(b"mdia", mdhd + box(b"minf", stbl))))
    assert H264Parameters.from_mp4(data).frame_rate == 15.0
