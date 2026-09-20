"""Per-clip resampling and RMS matching used by the paper's SVO-AQA runs."""
from __future__ import annotations

import array
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import List, Tuple

TARGET_SR = 48_000


def _read_pcm(path: Path) -> Tuple[List[float], int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        samples = array.array("h")
        samples.frombytes(raw)
        values = [sample / 32768.0 for sample in samples]
    elif width == 4:
        values = [
            struct.unpack("<i", raw[i : i + 4])[0] / 2147483648.0
            for i in range(0, len(raw), 4)
        ]
    else:
        raise ValueError(f"unsupported sample width {width}: {path}")
    return values, channels, rate


def _write_pcm(path: Path, values: List[float], channels: int, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = [max(-1.0, min(1.0, value)) for value in values]
    pcm = array.array("h", (int(round(value * 32767.0)) for value in clipped))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _rms(values: List[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate()


def resample_wav(src: Path, out: Path, target_sr: int = TARGET_SR) -> int:
    """Resample a WAV file to mono signed 16-bit PCM using ffmpeg."""
    source_sr = _sample_rate(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if source_sr == target_sr:
        shutil.copy2(src, out)
        return source_sr
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(out),
        ],
        check=True,
    )
    return source_sr


def match_rms_to_reference(
    src: Path,
    reference: Path,
    out: Path,
    *,
    peak_limit: float = 0.99,
    target_sr: int = TARGET_SR,
) -> dict:
    """Resample both clips, match source RMS to the reference, and peak-limit."""
    tmp_src = out.with_suffix(".source.wav")
    tmp_ref = out.with_suffix(".reference.wav")
    try:
        source_sr = resample_wav(src, tmp_src, target_sr)
        reference_sr = resample_wav(reference, tmp_ref, target_sr)
        source, channels, rate = _read_pcm(tmp_src)
        target, _, _ = _read_pcm(tmp_ref)
        source_rms = _rms(source)
        reference_rms = _rms(target)
        gain = reference_rms / source_rms if source_rms > 1e-10 else 1.0
        scaled = [value * gain for value in source]
        peak = max((abs(value) for value in scaled), default=0.0)
        peak_scale = min(1.0, peak_limit / peak) if peak else 1.0
        if peak_scale < 1.0:
            scaled = [value * peak_scale for value in scaled]
        _write_pcm(out, scaled, channels, rate)
    finally:
        for path in (tmp_src, tmp_ref):
            path.unlink(missing_ok=True)
    return {
        "target_sr": target_sr,
        "source_sr": source_sr,
        "reference_sr": reference_sr,
        "source_rms": round(source_rms, 6),
        "reference_rms": round(reference_rms, 6),
        "gain": round(gain, 6),
        "peak_scale": round(peak_scale, 6),
        "effective_gain": round(gain * peak_scale, 6),
        "output_rms": round(_rms(scaled), 6),
    }


def cached_rms_match(src: Path, reference: Path, cache_path: Path) -> Tuple[Path, dict | None]:
    rebuild = (
        not cache_path.is_file()
        or src.stat().st_mtime > cache_path.stat().st_mtime
        or reference.stat().st_mtime > cache_path.stat().st_mtime
    )
    if not rebuild:
        try:
            rebuild = _sample_rate(cache_path) != TARGET_SR
        except wave.Error:
            rebuild = True
    metadata = match_rms_to_reference(src, reference, cache_path) if rebuild else None
    return cache_path, metadata
