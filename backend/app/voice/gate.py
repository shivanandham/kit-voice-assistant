"""Server-side speech gate (ABU11): forward mic audio to Deepgram only while it holds speech.

A cheap energy gate on 16 kHz PCM16: frame RMS against an adaptive noise floor. The floor drops
at once to quieter audio and rises slowly towards louder audio, so constant noise (a fan, a
hum, a client streaming noise) lifts the floor and cannot hold the gate open, while speech,
which comes in bursts, stays well above it. Thresholds err towards letting audio through.
"""

from __future__ import annotations

import math
import struct
from collections import deque

BYTES_PER_SECOND = 32_000
GATE_OPEN_FACTOR = 3.0  # speech is this many times the noise floor's RMS
GATE_MIN_RMS = 150.0  # below this nothing counts as speech, whatever the floor
GATE_OPEN_S = 0.06  # this much consecutive loud audio opens the gate
GATE_PREROLL_S = 0.4  # audio kept from before the opening, so first syllables are not lost
GATE_HANGOVER_S = 1.2  # trailing audio after the last loud frame, for Deepgram's endpointing
GATE_FLOOR_RISE_S = 4.0  # time constant of the floor rising towards louder audio


def rms(pcm: bytes) -> float:
    """Root mean square of little-endian PCM16 samples."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


class EnergyGate:
    """Decides frame by frame what to forward; time is measured in audio, not wall clock."""

    def __init__(self) -> None:
        """Start closed, with an unknown (low) noise floor."""
        self.open = False
        self._floor = GATE_MIN_RMS / GATE_OPEN_FACTOR
        self._loud_s = 0.0
        self._quiet_s = 0.0
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0

    def _track_floor(self, level: float, seconds: float) -> None:
        if level < self._floor:
            self._floor = max(level, 1.0)
        else:
            self._floor += (level - self._floor) * min(1.0, seconds / GATE_FLOOR_RISE_S)

    def feed(self, pcm: bytes) -> list[bytes]:
        """Take one mic frame; return the audio to forward now (possibly with pre-roll)."""
        pcm = pcm[: len(pcm) & ~1]
        if not pcm:
            return []
        seconds = len(pcm) / BYTES_PER_SECOND
        level = rms(pcm)
        loud = level >= max(GATE_MIN_RMS, self._floor * GATE_OPEN_FACTOR)
        self._track_floor(level, seconds)
        if self.open:
            self._quiet_s = 0.0 if loud else self._quiet_s + seconds
            if self._quiet_s > GATE_HANGOVER_S:
                self.open = False
                self._loud_s = 0.0
            return [pcm]
        self._loud_s = self._loud_s + seconds if loud else 0.0
        self._preroll.append(pcm)
        self._preroll_bytes += len(pcm)
        while self._preroll_bytes - len(self._preroll[0]) >= GATE_PREROLL_S * BYTES_PER_SECOND:
            self._preroll_bytes -= len(self._preroll.popleft())
        if self._loud_s < GATE_OPEN_S:
            return []
        self.open, self._quiet_s = True, 0.0
        out = list(self._preroll)
        self._preroll.clear()
        self._preroll_bytes = 0
        return out
