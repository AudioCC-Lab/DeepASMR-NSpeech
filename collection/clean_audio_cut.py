"""
按 timeline 粗段做 ruptures 精细分段（与仓库根目录 trial.ipynb 同源算法），
为每个 timeline 行生成若干子段：JSON 中 start_sec/end_sec 仍为 timeline 边界，
clean_start/clean_end 为 ruptures 保留窗的绝对时间；title/label 等从原属 timeline 段继承。

整轨 librosa.load 一次（mono=False，与 notebook 一致按左声道分析），再对各片段切片检测。

ruptures（PELT 等）仅在 CPU 上运行；GPU 可选用于加速 **MFCC/delta**（见 --torch-device）。
批量处理多个 id 时可用 --workers 多进程（每进程仍各自解码整轨）。

结果写入 segment_label_parse.json（按 ruptures 展开 segments；可选）；
可选 --write-timeline 在 timeline.csv 追加 clean_start、clean_end（多子段时用分号分隔）。
可选 --skip-existing：若 JSON 中每条 segment 的必需键均存在且非 null，且 clean_end>clean_start，则跳过（不解码）。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler, normalize

_code_dir = Path(__file__).resolve().parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))


def _keyframe_relative(video_id: str, start: float, end: float) -> Optional[str]:
    try:
        from yt_frame_extract import segment_output_png_name  # noqa: WPS433
    except ImportError:
        return None
    name = segment_output_png_name(video_id, start, end)
    if not name:
        return None
    return str(Path("frames") / name)


def _extract_asmr_features_torch(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    n_mfcc: int,
    include_extra: bool,
    torch_device: str,
) -> np.ndarray:
    """在 torch_device 上计算 MFCC/delta/delta2；对比度等仍在 CPU 上用 librosa（与纯 librosa 谱略有数值差）。"""
    import librosa
    import torch
    import torchaudio

    device = torch.device(torch_device)
    w = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device).unsqueeze(0)
    mfcc_tx = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": 128,
            "center": True,
            "power": 2.0,
        },
    ).to(device)
    mfcc_t = mfcc_tx(w)[0]
    d1_t = torchaudio.functional.compute_deltas(mfcc_t)
    d2_t = torchaudio.functional.compute_deltas(d1_t)

    mfcc = mfcc_t.detach().cpu().numpy()
    delta = d1_t.detach().cpu().numpy()
    delta2 = d2_t.detach().cpu().numpy()

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    features = [mfcc, delta, delta2, contrast, flatness, zcr, rms]

    if include_extra:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        )
        stft = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        flux = np.sqrt(
            np.sum(np.diff(stft, axis=1, prepend=stft[:, :1]) ** 2, axis=0, keepdims=True)
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hf_mask = freqs >= 4000.0
        hf_energy = np.sum(stft[hf_mask, :] ** 2, axis=0, keepdims=True)
        total_energy = np.sum(stft**2, axis=0, keepdims=True) + 1e-8
        hf_ratio = hf_energy / total_energy
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)[None, :]
        features += [centroid, bandwidth, rolloff, flux, hf_ratio, onset]

    min_frames = min(feat.shape[1] for feat in features)
    features = [feat[:, :min_frames] for feat in features]
    return np.vstack(features)


def extract_asmr_features(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mfcc: int = 20,
    include_extra: bool = True,
    *,
    torch_device: Optional[str] = None,
) -> np.ndarray:
    import librosa

    if torch_device:
        return _extract_asmr_features_torch(
            y,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
            include_extra=include_extra,
            torch_device=torch_device,
        )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    features = [mfcc, delta, delta2, contrast, flatness, zcr, rms]

    if include_extra:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
        )
        stft = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        flux = np.sqrt(
            np.sum(np.diff(stft, axis=1, prepend=stft[:, :1]) ** 2, axis=0, keepdims=True)
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hf_mask = freqs >= 4000.0
        hf_energy = np.sum(stft[hf_mask, :] ** 2, axis=0, keepdims=True)
        total_energy = np.sum(stft**2, axis=0, keepdims=True) + 1e-8
        hf_ratio = hf_energy / total_energy
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)[None, :]
        features += [centroid, bandwidth, rolloff, flux, hf_ratio, onset]

    min_frames = min(feat.shape[1] for feat in features)
    features = [feat[:, :min_frames] for feat in features]
    return np.vstack(features)


def robust_standardize_frame_features(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.median(features, axis=1, keepdims=True)
    q25 = np.percentile(features, 25, axis=1, keepdims=True)
    q75 = np.percentile(features, 75, axis=1, keepdims=True)
    iqr = q75 - q25
    return (features - med) / (iqr + eps)


def sliding_mean_std_embeddings(
    features: np.ndarray,
    window_frames: int,
    step_frames: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    num_features, num_frames = features.shape
    if num_frames < window_frames:
        return np.array([], dtype=int), np.empty((0, 2 * num_features))

    starts = np.arange(0, num_frames - window_frames + 1, step_frames)
    ends = starts + window_frames
    center_frames = (starts + ends) // 2

    x = features
    csum = np.concatenate([np.zeros((num_features, 1)), np.cumsum(x, axis=1)], axis=1)
    csum2 = np.concatenate([np.zeros((num_features, 1)), np.cumsum(x * x, axis=1)], axis=1)
    sums = csum[:, ends] - csum[:, starts]
    sums2 = csum2[:, ends] - csum2[:, starts]
    mean = sums / window_frames
    mean2 = sums2 / window_frames
    var = np.maximum(mean2 - mean * mean, 0.0)
    std = np.sqrt(var + eps)
    embeddings = np.concatenate([mean, std], axis=0).T
    return center_frames, embeddings


def merge_short_segments_by_boundaries(boundaries: Sequence[int], min_segment_windows: int) -> List[int]:
    boundaries = sorted(set(int(x) for x in boundaries))
    if len(boundaries) <= 2:
        return boundaries

    changed = True
    while changed:
        changed = False
        new_boundaries = [boundaries[0]]
        i = 1
        while i < len(boundaries):
            prev = new_boundaries[-1]
            curr = boundaries[i]
            seg_len = curr - prev
            if i < len(boundaries) - 1 and seg_len < min_segment_windows:
                changed = True
                i += 1
            else:
                new_boundaries.append(curr)
                i += 1
        boundaries = new_boundaries
    return boundaries


def detect_change_points_ruptures(
    embeddings: np.ndarray,
    method: str = "pelt",
    model: str = "rbf",
    penalty: float = 20.0,
    n_bkps: Optional[int] = None,
    min_segment_windows: int = 5,
    jump: int = 1,
) -> List[int]:
    if len(embeddings) < 2 * min_segment_windows:
        return [0, len(embeddings)]

    x = np.asarray(embeddings, dtype=np.float64)

    if method == "pelt":
        algo = rpt.Pelt(model=model, min_size=min_segment_windows, jump=jump).fit(x)
        bkps = algo.predict(pen=penalty)
    elif method == "binseg":
        if n_bkps is None:
            raise ValueError("n_bkps must be provided when method='binseg'.")
        algo = rpt.Binseg(model=model, min_size=min_segment_windows, jump=jump).fit(x)
        bkps = algo.predict(n_bkps=n_bkps)
    elif method == "kernel":
        algo = rpt.KernelCPD(kernel=model, min_size=min_segment_windows, jump=jump).fit(x)
        bkps = algo.predict(n_bkps=n_bkps) if n_bkps is not None else algo.predict(pen=penalty)
    else:
        raise ValueError("method must be 'pelt', 'binseg', or 'kernel'.")

    boundaries = [0] + list(bkps)
    if boundaries[-1] != len(embeddings):
        boundaries.append(len(embeddings))
    boundaries = [b for b in boundaries if 0 <= b <= len(embeddings)]
    boundaries = sorted(set(boundaries))
    return merge_short_segments_by_boundaries(boundaries, min_segment_windows=min_segment_windows)


def seconds_to_hms(seconds: Optional[float], keep_ms: bool = True) -> str:
    if seconds is None:
        return ""
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if keep_ms:
        return f"{sign}{hours}:{minutes:02d}:{secs:06.3f}"
    return f"{sign}{hours}:{minutes:02d}:{int(round(secs)):02d}"


def add_hms_columns(df: pd.DataFrame, global_start: Optional[float] = None) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0:
        return df
    if "start" in df.columns:
        df["start_hms"] = df["start"].apply(seconds_to_hms)
    if "end" in df.columns:
        df["end_hms"] = df["end"].apply(seconds_to_hms)
    if "duration" in df.columns:
        df["duration_hms"] = df["duration"].apply(seconds_to_hms)
    if global_start is not None:
        if "start" in df.columns:
            df["start_rel"] = df["start"] - global_start
            df["start_rel_hms"] = df["start_rel"].apply(seconds_to_hms)
        if "end" in df.columns:
            df["end_rel"] = df["end"] - global_start
            df["end_rel_hms"] = df["end_rel"].apply(seconds_to_hms)
    return df


def boundaries_to_segments_with_gap(
    boundaries: Sequence[int],
    center_times: np.ndarray,
    global_start: float,
    global_end: float,
    min_keep_sec: float = 10.0,
    transition_gap_sec: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    n = len(center_times)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    boundaries = sorted(set(int(b) for b in boundaries))
    boundaries = [b for b in boundaries if 0 <= b <= n]
    if len(boundaries) == 0 or boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != n:
        boundaries.append(n)

    base_time_boundaries: List[float] = []
    for b in boundaries:
        if b <= 0:
            t = 0.0
        elif b >= n:
            t = global_end - global_start
        else:
            t = 0.5 * (center_times[b - 1] + center_times[b])
        base_time_boundaries.append(float(t))

    half_gap = max(0.0, float(transition_gap_sec)) / 2.0
    for raw_id, (left_b, right_b, left_w, right_w) in enumerate(
        zip(base_time_boundaries[:-1], base_time_boundaries[1:], boundaries[:-1], boundaries[1:])
    ):
        if raw_id == 0:
            t0_rel = left_b
        else:
            t0_rel = left_b + half_gap
        if raw_id == len(base_time_boundaries) - 2:
            t1_rel = right_b
        else:
            t1_rel = right_b - half_gap
        t0_rel = max(0.0, t0_rel)
        t1_rel = min(global_end - global_start, t1_rel)
        abs_start = global_start + t0_rel
        abs_end = global_start + t1_rel
        duration = abs_end - abs_start
        record: Dict[str, Any] = {
            "raw_segment_id": raw_id,
            "start": float(abs_start),
            "end": float(abs_end),
            "duration": float(duration),
            "left_window": int(left_w),
            "right_window": int(right_w),
            "transition_gap_sec": float(transition_gap_sec),
        }
        if duration >= min_keep_sec:
            record["segment_id"] = len(kept)
            record["kept"] = True
            kept.append(record)
        else:
            record["kept"] = False
            record["discard_reason"] = (
                f"duration < min_keep_sec ({min_keep_sec}) after transition_gap_sec ({transition_gap_sec})"
            )
            discarded.append(record)

    kept_df = add_hms_columns(pd.DataFrame(kept), global_start=global_start)
    discarded_df = add_hms_columns(pd.DataFrame(discarded), global_start=global_start)
    return kept_df, discarded_df


def segment_waveform_single_ear_with_ruptures(
    y_ch: np.ndarray,
    sr: int,
    global_start: float,
    global_end: float,
    *,
    n_fft: int = 1024,
    hop_length: int = 128,
    n_mfcc: int = 20,
    include_extra_features: bool = True,
    torch_device: Optional[str] = None,
    window_sec: float = 4.0,
    step_sec: float = 0.50,
    method: str = "pelt",
    model: str = "rbf",
    penalty: float = 20.0,
    n_bkps: Optional[int] = None,
    min_segment_sec: float = 15.0,
    min_keep_sec: float = 10.0,
    transition_gap_sec: float = 2.0,
    jump: int = 1,
    normalize_embeddings: bool = True,
    scale_embeddings: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """与 trial.ipynb `segment_audio_single_ear_with_ruptures` 一致，输入为已裁好的单声道波形。"""
    assert global_end > global_start
    if len(y_ch) == 0:
        raise ValueError("empty segment")

    y_ch = np.asarray(y_ch, dtype=np.float32)
    y_ch = y_ch - np.mean(y_ch)

    import librosa

    frame_features = extract_asmr_features(
        y=y_ch,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        include_extra=include_extra_features,
        torch_device=torch_device,
    )
    frame_features = robust_standardize_frame_features(frame_features)
    frame_dt = hop_length / sr
    window_frames = max(2, int(round(window_sec / frame_dt)))
    step_frames = max(1, int(round(step_sec / frame_dt)))

    center_frames, embeddings = sliding_mean_std_embeddings(
        frame_features,
        window_frames=window_frames,
        step_frames=step_frames,
    )
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings. Try smaller window_sec.")

    center_times = librosa.frames_to_time(center_frames, sr=sr, hop_length=hop_length)
    x = embeddings.copy()
    if scale_embeddings:
        x = StandardScaler().fit_transform(x)
    if normalize_embeddings:
        x = normalize(x)

    min_segment_windows = max(1, int(round(min_segment_sec / step_sec)))
    boundaries = detect_change_points_ruptures(
        embeddings=x,
        method=method,
        model=model,
        penalty=penalty,
        n_bkps=n_bkps,
        min_segment_windows=min_segment_windows,
        jump=jump,
    )
    kept_df, discarded_df = boundaries_to_segments_with_gap(
        boundaries=boundaries,
        center_times=center_times,
        global_start=global_start,
        global_end=global_end,
        min_keep_sec=min_keep_sec,
        transition_gap_sec=transition_gap_sec,
    )
    debug = {
        "global_start": float(global_start),
        "global_end": float(global_end),
        "sr": int(sr),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "method": method,
        "model": model,
        "penalty": float(penalty),
        "n_bkps": n_bkps,
        "min_segment_sec": float(min_segment_sec),
        "min_keep_sec": float(min_keep_sec),
        "transition_gap_sec": float(transition_gap_sec),
        "jump": int(jump),
        "hop_length": int(hop_length),
        "boundaries": boundaries,
    }
    return kept_df, discarded_df, debug


def pick_left_channel_slice(y_full: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """y_full: mono [samples] 或 stereo [channels, samples]。"""
    i0 = max(0, int(np.floor(start * sr)))
    i1 = min(y_full.shape[-1], int(np.ceil(end * sr)))
    if i1 <= i0:
        return np.array([], dtype=np.float32)
    if y_full.ndim == 1:
        seg = y_full[i0:i1]
    else:
        seg = y_full[0, i0:i1]
    return np.asarray(seg, dtype=np.float32)


def _field_map(fieldnames: List[str]) -> Dict[str, str]:
    return {(h or "").strip(): h for h in fieldnames if h}


def read_timeline_segments_with_rows(
    timeline_csv: Path,
    folder_video_id: str,
) -> Tuple[List[str], List[Dict[str, str]], List[Tuple[str, float, float, str]]]:
    need = frozenset({"video_id", "start", "end"})
    with timeline_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("timeline 无表头")
        fieldnames = list(r.fieldnames)
        fn = _field_map(fieldnames)
        hset = set(fn.keys())
        if not need.issubset(hset):
            raise ValueError("timeline.csv 需含 video_id、start、end")
        st_key = "segment_title" if "segment_title" in fn else None
        rows_raw: List[Dict[str, str]] = []
        segs: List[Tuple[str, float, float, str]] = []
        for row in r:
            if not isinstance(row, dict):
                continue
            vid = (row.get(fn["video_id"], "") or "").strip() or folder_video_id
            st = (row.get(fn["start"], "") or "").strip()
            en = (row.get(fn["end"], "") or "").strip()
            if not st or not en:
                continue
            seg_title = ""
            if st_key:
                seg_title = (row.get(fn[st_key], "") or "").strip()
            rows_raw.append(dict(row))
            segs.append((vid, float(st), float(en), seg_title))
    return fieldnames, rows_raw, segs


def find_audio_in_id_dir(id_dir: Path, video_id: str) -> Optional[Path]:
    for ext in (".wav", ".m4a", ".mp3", ".opus", ".webm", ".flac"):
        p = id_dir / f"{video_id}{ext}"
        if p.is_file():
            return p
    skip = {".csv", ".json", ".png", ".jpg", ".jpeg", ".txt"}
    cands = sorted(id_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if p.is_file() and p.suffix.lower() not in skip:
            return p
    return None


def find_parent_json_segment(
    segs: Any,
    start: float,
    end: float,
    *,
    eps: float = 0.51,
) -> Optional[Dict[str, Any]]:
    matches = match_json_segments_for_timeline(segs, start, end, eps_sec=eps)
    return matches[0] if matches else None


_SKIP_SEGMENT_REQUIRED_KEYS = frozenset(
    {"video_id", "start_sec", "end_sec", "clean_start", "clean_end"},
)


def should_skip_existing_clean_cut(json_path: Path) -> Tuple[bool, str]:
    """
    增量跳过：JSON 可读、segments 非空；每条含必需键且值非 null；
    且 clean_start/clean_end 可转为 float 且 clean_end > clean_start（不校验 timeline、clean_error 等）。
    """
    if not json_path.is_file():
        return False, "no segment json"
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "json unreadable"

    if not isinstance(payload, dict):
        return False, "invalid json root"

    jsegs = payload.get("segments")
    if not isinstance(jsegs, list) or not jsegs:
        return False, "no segments in json"

    for i, seg in enumerate(jsegs):
        if not isinstance(seg, dict):
            return False, f"segments[{i}] is not an object"
        missing = sorted(_SKIP_SEGMENT_REQUIRED_KEYS - set(seg.keys()))
        if missing:
            return False, f"segments[{i}] missing keys: {missing}"
        for k in _SKIP_SEGMENT_REQUIRED_KEYS:
            if seg.get(k) is None:
                return False, f"segments[{i}] {k} is null"
        try:
            cs = float(seg["clean_start"])
            ce = float(seg["clean_end"])
        except (TypeError, ValueError):
            return False, f"segments[{i}] clean_start/clean_end not numeric"
        if ce <= cs:
            return False, f"segments[{i}] clean_end <= clean_start"

    return True, "incremental skip"


def match_json_segments_for_timeline(
    jsegs: Any,
    start_sec: float,
    end_sec: float,
    *,
    eps_sec: float = 0.51,
) -> List[Dict[str, Any]]:
    """与 timeline 一行对齐的所有 JSON 段（同一 start_sec/end_sec，多 ruptures 子段）。"""
    if not isinstance(jsegs, list):
        return []
    out: List[Dict[str, Any]] = []
    for js in jsegs:
        if not isinstance(js, dict):
            continue
        jst, jen = js.get("start_sec"), js.get("end_sec")
        if jst is None or jen is None:
            continue
        try:
            if abs(float(jst) - float(start_sec)) <= eps_sec and abs(float(jen) - float(end_sec)) <= eps_sec:
                out.append(js)
        except (TypeError, ValueError):
            continue

    def _sort_key(d: Dict[str, Any]) -> Tuple[float, float]:
        try:
            k0 = float(d.get("clean_segment_index", 0))
        except (TypeError, ValueError):
            k0 = 0.0
        try:
            cs = d.get("clean_start")
            k1 = float(cs) if cs is not None else 0.0
        except (TypeError, ValueError):
            k1 = 0.0
        return (k0, k1)

    out.sort(key=_sort_key)
    return out


def write_segment_label_parse_ruptures(
    json_path: Path,
    video_id: str,
    new_segments: List[Dict[str, Any]],
    *,
    base_payload: Optional[Dict[str, Any]] = None,
    dry_run: bool,
    timeline_row_count: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = dict(base_payload) if isinstance(base_payload, dict) else {}
    for drop in ("segments", "segment_count", "clean_audio_method"):
        payload.pop(drop, None)

    payload["video_id"] = video_id
    payload.setdefault("source_timeline", "timeline.csv")
    for s in new_segments:
        if isinstance(s, dict):
            s.pop("clean_audio_debug", None)
    payload["segments"] = new_segments
    payload["segment_count"] = len(new_segments)
    payload["clean_audio_method"] = "ruptures_single_ear_trial_notebook"
    if timeline_row_count is not None:
        payload["timeline_csv_row_count"] = int(timeline_row_count)

    if not dry_run:
        with json_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")


def write_timeline_clean_columns(
    timeline_csv: Path,
    rows_raw: List[Dict[str, str]],
    fieldnames: List[str],
    clean_pairs_per_row: List[List[Tuple[float, float]]],
    *,
    dry_run: bool,
) -> None:
    extra = ("clean_start", "clean_end")
    out_fields = list(fieldnames)
    for c in extra:
        if c not in out_fields:
            out_fields.append(c)

    idx = 0
    new_rows: List[Dict[str, str]] = []
    for row in rows_raw:
        nr = dict(row)
        pairs = clean_pairs_per_row[idx] if idx < len(clean_pairs_per_row) else []
        idx += 1
        if len(pairs) == 1:
            cs, ce = pairs[0]
            nr["clean_start"] = str(cs)
            nr["clean_end"] = str(ce)
        elif len(pairs) > 1:
            nr["clean_start"] = ";".join(str(p[0]) for p in pairs)
            nr["clean_end"] = ";".join(str(p[1]) for p in pairs)
        else:
            nr.setdefault("clean_start", "")
            nr.setdefault("clean_end", "")
        for k in out_fields:
            nr.setdefault(k, "")
        new_rows.append(nr)

    if dry_run:
        return

    with timeline_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for nr in new_rows:
            w.writerow({k: nr.get(k, "") for k in out_fields})


def _sort_kept_df(kept_df: pd.DataFrame) -> pd.DataFrame:
    if len(kept_df) == 0:
        return kept_df
    if "segment_id" in kept_df.columns:
        return kept_df.sort_values(by="segment_id", kind="mergesort").reset_index(drop=True)
    return kept_df.sort_values(by="start", kind="mergesort").reset_index(drop=True)


def build_child_segment_dict(
    *,
    video_id: str,
    timeline_start: float,
    timeline_end: float,
    clean_start: float,
    clean_end: float,
    clean_segment_index: int,
    label: str,
    parent: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if parent:
        seg = copy.deepcopy(parent)
    else:
        seg = {}

    for drop_k in (
        "clean_audio_debug",
        "clean_error",
        "clean_detected_start",
        "clean_detected_end",
    ):
        seg.pop(drop_k, None)

    seg["video_id"] = video_id
    seg["start_sec"] = float(timeline_start)
    seg["end_sec"] = float(timeline_end)
    seg["label"] = (label or "").strip() or seg.get("label")
    seg["clean_start"] = float(clean_start)
    seg["clean_end"] = float(clean_end)
    seg["clean_segment_index"] = int(clean_segment_index)
    seg["clean_detected"] = True
    kf = _keyframe_relative(video_id, clean_start, clean_end)
    if kf is not None:
        seg["keyframe_relative"] = kf
    return seg


def process_one_video(
    video_id: str,
    id_dir: Path,
    *,
    sr_target: Optional[float],
    dry_run: bool,
    skip_existing: bool,
    write_timeline: bool,
    segment_json_name: str,
    kwargs_ruptures: Dict[str, Any],
    audio_file: Optional[Path] = None,
) -> Tuple[int, Optional[str], bool]:
    tl = id_dir / "timeline.csv"
    if not tl.is_file():
        return 0, "no timeline.csv", False

    try:
        timeline_fields, rows_raw, segs = read_timeline_segments_with_rows(tl, video_id)
    except Exception as e:
        return 0, str(e), False

    if not segs:
        return 0, "no valid timeline rows", False

    json_path = id_dir / segment_json_name
    if dry_run:
        would_skip = False
        if skip_existing:
            would_skip, _ = should_skip_existing_clean_cut(json_path)
        return len(segs), None, would_skip

    if skip_existing:
        do_skip, _skip_reason = should_skip_existing_clean_cut(json_path)
        if do_skip:
            return len(segs), None, True

    if audio_file is not None:
        audio_path = audio_file.resolve()
        if not audio_path.is_file():
            return 0, f"audio file not found: {audio_path}", False
    else:
        audio_path = find_audio_in_id_dir(id_dir, video_id)
        if audio_path is None:
            return 0, "no audio file", False

    import librosa

    y_full, sr = librosa.load(str(audio_path), sr=sr_target, mono=False)
    duration = y_full.shape[-1] / sr

    old_payload: Dict[str, Any] = {}
    if json_path.is_file():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                old_payload = loaded
        except (OSError, json.JSONDecodeError):
            old_payload = {}
    old_segs = old_payload.get("segments")
    if not isinstance(old_segs, list):
        old_segs = []

    out_segments: List[Dict[str, Any]] = []
    clean_pairs_per_row: List[List[Tuple[float, float]]] = []

    for vid, start, end, label in segs:
        parent = find_parent_json_segment(old_segs, start, end)

        if end > duration + 0.5:
            seg = build_child_segment_dict(
                video_id=vid,
                timeline_start=start,
                timeline_end=end,
                clean_start=start,
                clean_end=end,
                clean_segment_index=0,
                label=label,
                parent=parent,
            )
            seg["clean_detected"] = False
            seg["clean_error"] = "end beyond audio duration"
            out_segments.append(seg)
            clean_pairs_per_row.append([(start, end)])
            continue

        y_seg = pick_left_channel_slice(y_full, sr, start, end)
        seg_dur = end - start
        if seg_dur < 1.0 or len(y_seg) < 256:
            seg = build_child_segment_dict(
                video_id=vid,
                timeline_start=start,
                timeline_end=end,
                clean_start=start,
                clean_end=end,
                clean_segment_index=0,
                label=label,
                parent=parent,
            )
            seg["clean_detected"] = False
            seg["clean_error"] = "segment too short for ruptures"
            out_segments.append(seg)
            clean_pairs_per_row.append([(start, end)])
            continue

        try:
            kept_df, _, _ = segment_waveform_single_ear_with_ruptures(
                y_seg,
                sr,
                start,
                end,
                **kwargs_ruptures,
            )
        except Exception as e:
            seg = build_child_segment_dict(
                video_id=vid,
                timeline_start=start,
                timeline_end=end,
                clean_start=start,
                clean_end=end,
                clean_segment_index=0,
                label=label,
                parent=parent,
            )
            seg["clean_detected"] = False
            seg["clean_error"] = str(e)
            out_segments.append(seg)
            clean_pairs_per_row.append([(start, end)])
            continue

        kept_df = _sort_kept_df(kept_df)
        row_pairs: List[Tuple[float, float]] = []

        if len(kept_df) == 0:
            seg = build_child_segment_dict(
                video_id=vid,
                timeline_start=start,
                timeline_end=end,
                clean_start=start,
                clean_end=end,
                clean_segment_index=0,
                label=label,
                parent=parent,
            )
            seg["clean_detected"] = False
            seg["clean_error"] = "ruptures produced no kept segments"
            out_segments.append(seg)
            clean_pairs_per_row.append([(start, end)])
            continue

        for j, (_, row) in enumerate(kept_df.iterrows()):
            cs_f = float(row["start"])
            ce_f = float(row["end"])
            row_pairs.append((cs_f, ce_f))
            rid_raw = row["segment_id"] if "segment_id" in row else j
            rid = int(rid_raw) if not pd.isna(rid_raw) else j
            seg = build_child_segment_dict(
                video_id=vid,
                timeline_start=start,
                timeline_end=end,
                clean_start=cs_f,
                clean_end=ce_f,
                clean_segment_index=j,
                label=label,
                parent=parent,
            )
            seg["ruptures_kept_segment_id"] = rid
            out_segments.append(seg)

        clean_pairs_per_row.append(row_pairs)

    write_segment_label_parse_ruptures(
        json_path,
        video_id,
        out_segments,
        base_payload=old_payload,
        dry_run=dry_run,
        timeline_row_count=len(segs),
    )

    if write_timeline and timeline_fields and not dry_run:
        write_timeline_clean_columns(tl, rows_raw, timeline_fields, clean_pairs_per_row, dry_run=False)

    return len(segs), None, False


def process_one_video_packed(
    packed: Tuple[
        str,
        str,
        Optional[float],
        bool,
        bool,
        bool,
        str,
        Dict[str, Any],
        Optional[str],
    ],
) -> Tuple[str, int, Optional[str], bool]:
    """供多进程调用；packed 含 skip_existing。"""
    (
        video_id,
        id_dir_str,
        sr_target,
        dry_run,
        skip_existing,
        write_timeline,
        segment_json_name,
        kwargs_ruptures,
        audio_file_str,
    ) = packed
    id_dir = Path(id_dir_str)
    audio_override = Path(audio_file_str) if audio_file_str else None
    n, err, skipped = process_one_video(
        video_id,
        id_dir,
        sr_target=sr_target,
        dry_run=dry_run,
        skip_existing=skip_existing,
        write_timeline=write_timeline,
        segment_json_name=segment_json_name,
        kwargs_ruptures=kwargs_ruptures,
        audio_file=audio_override,
    )
    return video_id, n, err, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "按 trial.ipynb：timeline 段内 ruptures 精细分段，写入 segment_label_parse.json "
            "（start_sec/end_sec 为 timeline；clean_* 为 ruptures 窗）。"
            " 不传 --id-dir 时遍历 audio 下各 id 目录。"
        ),
    )
    ap.add_argument("--root", type=Path, default=_code_dir.parent, help="项目根目录")
    ap.add_argument(
        "--audio-dir",
        "--audio-folder",
        type=Path,
        default=None,
        dest="audio_dir",
        metavar="DIR",
        help="audio 根目录（默认 <root>/audio）",
    )
    ap.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        dest="audio_path_alias",
        metavar="DIR",
        help="同 --audio-dir",
    )
    ap.add_argument(
        "--id-dir",
        type=Path,
        default=None,
        help="只处理该目录（须含 timeline.csv）；可与 --audio-file 联用",
    )
    ap.add_argument(
        "--audio-file",
        type=Path,
        default=None,
        help="显式整轨音频路径；须与 --id-dir 同时使用",
    )
    ap.add_argument("--ids", type=str, default="", help="仅处理指定 id，逗号分隔")
    ap.add_argument("--dry-run", action="store_true", help="不写文件、不解码")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="若已有 JSON：每条 segment 上述键均非 null，且 clean_end>clean_start，则跳过（不解码）",
    )
    ap.add_argument(
        "--write-timeline",
        action="store_true",
        help="在 timeline.csv 追加 clean_start、clean_end（多子段用分号分隔）",
    )
    ap.add_argument(
        "--segment-json-name",
        type=str,
        default="segment_label_parse.json",
        help="写入的 JSON 文件名",
    )
    ap.add_argument("--sr", type=float, default=22050, help="分析用采样率 Hz（默认 22050，与 notebook 一致）")
    ap.add_argument("--hop-length", type=int, default=128)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--n-mfcc", type=int, default=20)
    ap.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="滑窗时长（秒），与 trial.ipynb 示例单元一致（函数默认 4，示例用 3）",
    )
    ap.add_argument("--step-sec", type=float, default=0.5, help="滑窗步长（秒）")
    ap.add_argument(
        "--ruptures-method",
        type=str,
        default="pelt",
        choices=("pelt", "binseg", "kernel"),
        help="ruptures 算法",
    )
    ap.add_argument("--ruptures-model", type=str, default="rbf", help="pelt/binseg: rbf/l2/linear；kernel: rbf/linear")
    ap.add_argument(
        "--penalty",
        type=float,
        default=15.0,
        help="pelt/kernel 惩罚（trial.ipynb 示例为 15；函数签名为 20）",
    )
    ap.add_argument("--n-bkps", type=int, default=None, help="binseg 或 kernel 固定变点数（可选）")
    ap.add_argument(
        "--min-segment-sec",
        type=float,
        default=10.0,
        help="ruptures 最小段长（秒），与 trial.ipynb 示例一致",
    )
    ap.add_argument("--min-keep-sec", type=float, default=10.0, help="transition gap 后最短保留段（秒）")
    ap.add_argument("--transition-gap-sec", type=float, default=2.0, help="变点处丢弃过渡带（秒）")
    ap.add_argument("--jump", type=int, default=1, help="ruptures jump")
    ap.add_argument("--no-extra-features", action="store_true", help="关闭 notebook 中的额外谱特征")
    ap.add_argument("--no-normalize-embeddings", action="store_true", help="不对嵌入做 sklearn normalize")
    ap.add_argument("--no-scale-embeddings", action="store_true", help="不对嵌入做 StandardScaler")
    ap.add_argument(
        "--torch-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="用 PyTorch/torchaudio 在该设备上算 MFCC+delta（如 cuda 或 cuda:0）；需 pip install torch torchaudio。不传则全程 librosa（CPU）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行处理的视频目录数（多进程，默认 1）。与 CUDA 同用注意显存，可适当减小",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    audio_dir = (args.audio_path_alias or args.audio_dir or (root / "audio")).resolve()

    if args.audio_file and not args.id_dir:
        print("[ERR] --audio-file 必须与 --id-dir 同时使用", file=sys.stderr)
        return 1

    id_dir_single: Optional[Path] = None
    if args.id_dir is not None:
        id_dir_single = args.id_dir.resolve()
        if not id_dir_single.is_dir():
            print(f"[ERR] --id-dir 不是目录: {id_dir_single}", file=sys.stderr)
            return 1
    elif not audio_dir.is_dir():
        print(f"[ERR] audio 目录不存在: {audio_dir}", file=sys.stderr)
        return 1

    audio_override: Optional[Path] = None
    if args.audio_file is not None:
        audio_override = args.audio_file.resolve()
        if not audio_override.is_file():
            print(f"[ERR] --audio-file 不是文件: {audio_override}", file=sys.stderr)
            return 1

    torch_dev = (args.torch_device or "").strip() or None
    if torch_dev:
        try:
            import torch
        except ImportError:
            print("[ERR] 使用 --torch-device 需要安装 torch、torchaudio：pip install torch torchaudio", file=sys.stderr)
            return 1
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            print("[ERR] 使用 --torch-device 需要 torchaudio：pip install torchaudio", file=sys.stderr)
            return 1
        if torch_dev.startswith("cuda") and not torch.cuda.is_available():
            ver = getattr(torch, "__version__", "?")
            print(
                "[ERR] 指定了 CUDA 但 PyTorch 未启用 GPU：torch.__version__="
                f"{ver}（若为 xxx+cpu 则是 CPU 专用包，需换装带 CUDA 的 torch/torchaudio）。"
                " 请打开 https://pytorch.org/get-started/locally/ 按环境选择 CUDA 版 pip 命令重装。",
                file=sys.stderr,
            )
            return 1

    if args.workers > 1 and torch_dev and torch_dev.startswith("cuda"):
        print(
            "[WARN] 多进程与 CUDA 同时使用可能占满显存；若 OOM 请改用 --workers 1",
            file=sys.stderr,
        )

    kwargs_ruptures: Dict[str, Any] = {
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mfcc": args.n_mfcc,
        "include_extra_features": not bool(args.no_extra_features),
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "method": args.ruptures_method,
        "model": args.ruptures_model,
        "penalty": float(args.penalty),
        "n_bkps": args.n_bkps,
        "min_segment_sec": float(args.min_segment_sec),
        "min_keep_sec": float(args.min_keep_sec),
        "transition_gap_sec": float(args.transition_gap_sec),
        "jump": int(args.jump),
        "normalize_embeddings": not bool(args.no_normalize_embeddings),
        "scale_embeddings": not bool(args.no_scale_embeddings),
        "torch_device": torch_dev,
    }

    id_filter: Optional[set[str]] = None
    if (args.ids or "").strip():
        id_filter = {x.strip() for x in args.ids.split(",") if x.strip()}

    if id_dir_single is not None:
        subdirs = [id_dir_single]
    else:
        subdirs = sorted([p for p in audio_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())

    tasks: List[
        Tuple[str, str, Optional[float], bool, bool, bool, str, Dict[str, Any], Optional[str]]
    ] = []
    for d in subdirs:
        vid = d.name
        if id_filter is not None and vid not in id_filter:
            continue
        tasks.append(
            (
                vid,
                str(d.resolve()),
                args.sr,
                bool(args.dry_run),
                bool(args.skip_existing),
                bool(args.write_timeline),
                args.segment_json_name,
                kwargs_ruptures,
                str(audio_override) if audio_override is not None else None,
            )
        )

    ok = 0
    skipped_n = 0
    errs: List[str] = []
    max_workers = max(1, int(args.workers))
    if max_workers <= 1 or len(tasks) <= 1:
        for packed in tasks:
            vid, n, err, skipped = process_one_video_packed(packed)
            if err:
                errs.append(f"{vid}: {err}")
                continue
            if skipped:
                skipped_n += 1
                if args.dry_run:
                    print(f"[SKIP-DRY-RUN] {vid} timeline_rows={n}（增量将跳过）")
                else:
                    print(f"[SKIP] {vid} timeline_rows={n} incremental")
                continue
            ok += 1
            tag = "[DRY-RUN]" if args.dry_run else "[OK]"
            print(f"{tag} {vid} timeline_rows={n} json={Path(packed[1]) / args.segment_json_name}")
    else:
        n_workers = min(max_workers, len(tasks))
        vid_to_dir = {t[0]: Path(t[1]) for t in tasks}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_one_video_packed, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                vid_sched = futures[fut]
                try:
                    vid_out, n, err, skipped = fut.result()
                except Exception as e:
                    errs.append(f"{vid_sched}: {e}")
                    continue
                if err:
                    errs.append(f"{vid_out}: {err}")
                    continue
                if skipped:
                    skipped_n += 1
                    if args.dry_run:
                        print(f"[SKIP-DRY-RUN] {vid_out} timeline_rows={n}（增量将跳过）")
                    else:
                        print(f"[SKIP] {vid_out} timeline_rows={n} incremental")
                    continue
                ok += 1
                tag = "[DRY-RUN]" if args.dry_run else "[OK]"
                id_dir = vid_to_dir.get(vid_out, Path("."))
                print(f"{tag} {vid_out} timeline_rows={n} json={id_dir / args.segment_json_name}")

    print(f"完成: 已处理={ok}, 跳过={skipped_n}, 失败={len(errs)}")
    for e in errs[:40]:
        print(f"  {e}", file=sys.stderr)
    if len(errs) > 40:
        print(f"  ... 另有 {len(errs)-40} 条", file=sys.stderr)

    return 0 if not errs else 2


if __name__ == "__main__":
    raise SystemExit(main())
