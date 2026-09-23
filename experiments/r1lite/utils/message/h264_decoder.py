"""Stateful H.264 decoder for camera topics published in h264 mode.

The /signal_camera driver publishes CompressedImage with format="h264" and
has no reconfigurable JPEG/PNG alternative available to this client
(encoding_mode is read once at node startup from a config file outside this
repo, and ros2 param set on the live node does not change it). Unlike JPEG,
an H.264 byte stream is not self-contained per message — frames reference
prior frames (I/P-frame structure) — so decoding needs one persistent
decoder per camera stream, not a stateless per-message call.
"""

from __future__ import annotations

import logging

import av
import numpy as np

logger = logging.getLogger(__name__)


class H264StreamDecoder:
    """Feeds raw Annex-B H.264 byte chunks in and yields decoded RGB frames.

    One instance must be kept per camera topic (each is an independent
    encoded stream with its own SPS/PPS state).
    """

    def __init__(self):
        self._codec = av.CodecContext.create("h264", "r")

    def decode(self, data: bytes) -> np.ndarray | None:
        """Feed one message's bytes in; return the latest decoded frame as
        an RGB [C,H,W] uint8 array, or None if no full frame completed yet
        (e.g. only SPS/PPS or a partial NAL was fed so far)."""
        try:
            packets = self._codec.parse(data)
            frame = None
            for packet in packets:
                for f in self._codec.decode(packet):
                    frame = f
        except Exception:
            # Broad catch: PyAV's ffmpeg-backed errors must never take down
            # the ROS2 executor thread over one bad/partial chunk (e.g. a
            # P-frame arriving before the stream's first keyframe).
            logger.warning("H.264 decode error, dropping this chunk", exc_info=True)
            return None

        if frame is None:
            return None
        return frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
