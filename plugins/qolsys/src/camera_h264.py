"""Extract incremental H264 from IQ2's growing MP4 camera files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

_START_CODE = b"\x00\x00\x00\x01"
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
_MAX_NAL_BYTES = 4_000_000


def _boxes(data: bytes, start: int = 0, end: int | None = None) -> Iterator[tuple[bytes, int, int]]:
    end = len(data) if end is None else end
    while start + 8 <= end:
        size = int.from_bytes(data[start : start + 4], "big")
        kind = data[start + 4 : start + 8]
        header = 8
        if size == 1:
            if start + 16 > end:
                raise ValueError("Incomplete extended MP4 box header")
            size = int.from_bytes(data[start + 8 : start + 16], "big")
            header = 16
        if size == 0:
            size = end - start
        if size < header:
            raise ValueError("Invalid MP4 box size")
        box_end = start + size
        # IQ2 leaves a placeholder mdat size until recording closes.
        if box_end > end and kind != b"mdat":
            return
        yield kind, start + header, min(box_end, end)
        start = box_end


def _avcc(data: bytes, start: int = 0, end: int | None = None) -> bytes | None:
    for kind, payload, box_end in _boxes(data, start, end):
        if kind == b"avcC":
            return data[payload:box_end]
        if kind in _CONTAINERS:
            child = payload
        elif kind == b"stsd":
            child = payload + 8  # FullBox flags and entry count.
        elif kind in {b"avc1", b"avc3"}:
            child = payload + 78  # VisualSampleEntry fixed fields.
        else:
            continue
        found = _avcc(data, child, box_end)
        if found is not None:
            return found
    return None


def _find(data: bytes, kind: bytes, start: int = 0, end: int | None = None) -> tuple[int, int] | None:
    for found, payload, box_end in _boxes(data, start, end):
        if found == kind:
            return payload, box_end
        if found in _CONTAINERS:
            nested = _find(data, kind, payload, box_end)
            if nested is not None:
                return nested
    return None


def _frame_rate(data: bytes) -> float | None:
    """Average frame rate of the first track with both a timescale and sample timing."""
    moov = _find(data, b"moov")
    if moov is None:
        return None
    for kind, payload, box_end in _boxes(data, *moov):
        if kind != b"trak":
            continue
        mdhd, stts = _find(data, b"mdhd", payload, box_end), _find(data, b"stts", payload, box_end)
        if mdhd is None or stts is None or mdhd[1] - mdhd[0] < 24 or stts[1] - stts[0] < 8:
            continue
        # FullBox version selects 32- or 64-bit creation/modification times.
        offset = mdhd[0] + (20 if data[mdhd[0]] == 1 else 12)
        timescale = int.from_bytes(data[offset : offset + 4], "big")
        entries = int.from_bytes(data[stts[0] + 4 : stts[0] + 8], "big")
        frames = duration = 0
        for entry in range(min(entries, (stts[1] - stts[0] - 8) // 8)):
            at = stts[0] + 8 + entry * 8
            count = int.from_bytes(data[at : at + 4], "big")
            frames += count
            duration += count * int.from_bytes(data[at + 4 : at + 8], "big")
        if frames and duration and timescale:
            return frames * timescale / duration
    return None


@dataclass(frozen=True)
class H264Parameters:
    """Decoder configuration obtained from a previously completed native clip."""

    length_size: int
    parameter_sets: tuple[bytes, ...]
    frame_rate: float | None = None

    @classmethod
    def from_mp4(cls, data: bytes) -> H264Parameters:
        config = _avcc(data)
        if config is None or len(config) < 7 or config[0] != 1:
            raise ValueError("No valid AVC decoder configuration found")
        length_size = (config[4] & 3) + 1
        offset = 6
        parameter_sets: list[bytes] = []
        counts = [config[5] & 31]
        for expected_type in (7, 8):
            if expected_type == 8:
                if offset >= len(config):
                    raise ValueError("Missing AVC picture parameter count")
                counts.append(config[offset])
                offset += 1
            for _ in range(counts[-1]):
                if offset + 2 > len(config):
                    raise ValueError("Incomplete AVC parameter length")
                size = int.from_bytes(config[offset : offset + 2], "big")
                offset += 2
                if size == 0 or offset + size > len(config):
                    raise ValueError("Incomplete AVC parameter set")
                parameter = config[offset : offset + size]
                if parameter[0] & 31 != expected_type:
                    raise ValueError("Unexpected AVC parameter type")
                parameter_sets.append(parameter)
                offset += size
        if not all(counts):
            raise ValueError("Both SPS and PPS are required")
        return cls(length_size, tuple(parameter_sets), _frame_rate(data))


class GrowingH264Reader:
    """Read new complete NAL units from successive prefixes of one MP4 file.

    Create a new reader for each recording. Returned bytes are Annex B H264,
    suitable for a decoder/FFmpeg pipe. No earlier clip's frames are reused.
    """

    def __init__(self, parameters: H264Parameters) -> None:
        if parameters.length_size not in (1, 2, 3, 4):
            raise ValueError("Unsupported AVC NAL length size")
        self._parameters = parameters
        self._mdat_start: int | None = None
        self._offset = 0
        self._initialized = False

    def feed(self, data: bytes) -> bytes:
        payload = next((start for kind, start, _end in _boxes(data) if kind == b"mdat"), None)
        if payload is None:
            return b""
        if self._mdat_start is None:
            self._mdat_start = payload
            self._offset = payload
        elif payload != self._mdat_start or len(data) < self._offset:
            raise ValueError("Recording changed or truncated; use a new reader")
        output: list[bytes] = []
        width = self._parameters.length_size
        while self._offset + width <= len(data):
            size = int.from_bytes(data[self._offset : self._offset + width], "big")
            if size == 0:
                break  # Unwritten/preallocated bytes.
            if size > _MAX_NAL_BYTES:
                raise ValueError("H264 NAL exceeds the bounded packet size")
            end = self._offset + width + size
            if end > len(data):
                break
            nal = data[self._offset + width : end]
            if nal[0] & 128 or not 1 <= nal[0] & 31 <= 23:
                raise ValueError("Invalid H264 NAL header")
            if not self._initialized:
                output.extend(_START_CODE + parameter for parameter in self._parameters.parameter_sets)
                self._initialized = True
            output.append(_START_CODE + nal)
            self._offset = end
        return b"".join(output)
