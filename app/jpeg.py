"""Read JPEG dimensions without allocating a decoded image."""

from __future__ import annotations

SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Return width and height from a bounded JPEG header before entropy decoding.

    Only markers outside length-delimited segments are interpreted. Zero-height
    images (which defer their dimensions to a DNL marker) are not accepted.
    """
    if payload[:2] != b"\xff\xd8":
        raise ValueError("frame_format")
    offset = 2
    dimensions = None
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ValueError("frame_decode")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset == len(payload):
            break
        marker = payload[offset]
        offset += 1
        # Stuffed bytes, restart markers and another SOI cannot occur here.
        if marker == 0 or 0xD0 <= marker <= 0xD9:
            raise ValueError("frame_decode")
        if marker == 0x01:  # TEM is the only standalone header marker.
            continue
        if offset + 2 > len(payload):
            break
        length = int.from_bytes(payload[offset : offset + 2], "big")
        end = offset + length
        if length < 2 or end > len(payload):
            break
        if marker in SOF_MARKERS:
            if dimensions is not None or length < 8:
                raise ValueError("frame_decode")
            components = payload[offset + 7]
            if components == 0 or length != 8 + 3 * components:
                raise ValueError("frame_decode")
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            if width == 0 or height == 0:
                raise ValueError("frame_dimensions")
            dimensions = (width, height)
        elif marker == 0xDA:  # SOS: all dimensions must be known before decoding.
            if dimensions is None:
                raise ValueError("frame_decode")
            return dimensions
        offset = end
    raise ValueError("frame_decode")
