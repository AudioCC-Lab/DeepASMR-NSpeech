from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

from timeline_parse import parse_seconds_from_timestamp_token

REQUIRED_OLD = frozenset({"video_id", "time", "seconds", "segment_title"})
OUTPUT_FIELDS = ["video_id", "time", "start", "end", "seconds", "segment_title"]


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        rate = float(wf.getframerate())
        if rate <= 0:
            raise ValueError("invalid sample rate")
        return nframes / rate


def find_wav_in_dir(directory: Path) -> Path | None:
    vid = directory.name
    cand = directory / f"{vid}.wav"
    if cand.is_file():
        return cand
    wavs = sorted(directory.glob("*.wav"))
    return wavs[0] if len(wavs) == 1 else None


def fmt_seconds(value: float) -> str:
    if value != value or value == float("inf"):  # NaN / inf
        raise ValueError("invalid duration")
    iv = int(value)
    if abs(value - iv) < 1e-9:
        return str(iv)
    s = f"{value:.9f}".rstrip("0").rstrip(".")
    return s or "0"


def reconcile_seconds_from_time(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """
    用 time 列重算 canonical 秒数；与 CSV 中原 seconds 不一致时以 time 为准并计数。
    若 time 无法解析则回退沿用原 seconds（仅当仍为整数秒）。
    """
    out: list[dict[str, str]] = []
    n_corrected = 0
    for row in rows:
        t = (row.get("time") or "").strip()
        rec = parse_seconds_from_timestamp_token(t)
        stored_raw = (row.get("seconds") or "").strip()
        try:
            stored = int(float(stored_raw)) if stored_raw else None
        except ValueError:
            stored = None

        if rec is None:
            if stored is None:
                raise ValueError(f"无法从 time「{t}」解析秒数且无有效 seconds 列")
            canonical = stored
        else:
            canonical = rec
            if stored is not None and stored != rec:
                n_corrected += 1


        out.append(
            {
                **row,
                "seconds": str(canonical),
            }
        )
    return out, n_corrected


def sort_rows_by_seconds(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed = [(i, r) for i, r in enumerate(rows)]
    indexed.sort(
        key=lambda ir: (
            int(float(ir[1].get("seconds") or "0")),
            ir[0],
        )
    )
    return [ir[1] for ir in indexed]


def normalize_rows(
    rows: list[dict[str, str]],
    *,
    last_end_sec: float,
) -> list[dict[str, str]]:
    if not rows:
        return []

    starts: list[int] = []
    for row in rows:
        raw = (row.get("seconds") or "").strip()
        starts.append(int(float(raw)))

    out: list[dict[str, str]] = []
    n = len(starts)
    for i, row in enumerate(rows):
        s = starts[i]
        end_sec: float = float(starts[i + 1]) if i + 1 < n else float(last_end_sec)
        if end_sec < s:
            raise ValueError(f"segment end ({end_sec}s) before start ({s}s)")
        out.append(
            {
                "video_id": row.get("video_id", ""),
                "time": row.get("time", ""),
                "start": str(s),
                "end": fmt_seconds(end_sec),
                "seconds": (row.get("seconds") or "").strip(),
                "segment_title": row.get("segment_title", ""),
            }
        )
    return out


def normalize_rows_to_disk(
    path: Path,
    norm_rows_raw: list[dict[str, str]],
    *,
    dry_run: bool,
    msg_prefix: str = "",
) -> tuple[bool, str]:
    """
    对已整理好的 timeline 行（仅 video_id/time/seconds/segment_title）做 reconcile、排序、
    start/end、写回磁盘。msg_prefix 会拼进返回文案前段。
    """
    if not norm_rows_raw:
        return False, f"{msg_prefix}no data rows"

    wav_path = find_wav_in_dir(path.parent)
    if wav_path is None:
        return False, f"{msg_prefix}no .wav found in folder (need duration for last segment end)"

    try:
        dur = wav_duration_seconds(wav_path)
    except Exception as e:
        return False, f"{msg_prefix}cannot read wav duration: {e}"

    try:
        reconciled, n_rec = reconcile_seconds_from_time(norm_rows_raw)
        ordered = sort_rows_by_seconds(reconciled)
        out_rows = normalize_rows(ordered, last_end_sec=dur)
    except Exception as e:
        return False, f"{msg_prefix}{e}"

    extra = f", 按 time 校正 seconds {n_rec} 处" if n_rec else ""
    if dry_run:
        return True, f"{msg_prefix}dry-run ok ({len(out_rows)} rows, wav={wav_path.name}{extra})"

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    tmp.replace(path)
    return True, f"{msg_prefix}updated ({len(out_rows)} rows{extra})"


def process_file(
    path: Path,
    *,
    dry_run: bool,
) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.strip():
        return False, "empty file"

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return False, "no header"
        fn = {h.strip(): h for h in reader.fieldnames if h}
        missing = REQUIRED_OLD - set(fn.keys())
        if missing:
            return False, f"missing columns {sorted(missing)}"

        norm_rows_raw: list[dict[str, str]] = []
        for row in reader:
            if isinstance(row, dict):
                nr = {(k or "").strip(): (row.get(fn.get(k, k)) or "").strip() for k in REQUIRED_OLD}
                norm_rows_raw.append(
                    {
                        "video_id": nr.get("video_id", ""),
                        "time": nr.get("time", ""),
                        "seconds": nr.get("seconds", ""),
                        "segment_title": nr.get("segment_title", ""),
                    }
                )

    return normalize_rows_to_disk(path, norm_rows_raw, dry_run=dry_run)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="在每段 <audio-dir>/<id>/timeline.csv 中：先用 time 列按「先 H:M:S 再 M:S」解析并校验 seconds（"
        "与 time 不符则以 time 为准），再按时序排序，最后在 time 后写入 start/end（秒）；"
        "末段 end 取自同目录 wav 时长",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="元数据根目录；失败报告 timeline_normalize_errors.txt 默认写在此目录",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（默认：<root>/audio），须与下载 --audio-dir 一致",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查，不写回文件",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    audio = (args.audio_dir or (root / "audio")).resolve()
    if not audio.is_dir():
        raise SystemExit(f"audio 目录不存在: {audio}")

    files = sorted(audio.glob("*/timeline.csv"), key=lambda p: str(p.parent.name).casefold())
    ok, bad = 0, 0
    failures: list[tuple[str, str]] = []
    for p in files:
        rel = p.relative_to(audio)
        good, msg = process_file(p, dry_run=args.dry_run)
        tag = "[OK]" if good else "[SKIP]"
        print(f"{tag} {rel}: {msg}")
        if good:
            ok += 1
        else:
            bad += 1
            failures.append((str(rel).replace("\\", "/"), msg))
    suf = " (dry-run)" if args.dry_run else ""
    print(f"[SUMMARY]{suf} 成功 {ok}, 跳过/失败 {bad}, 合计 {len(files)}")
    err_path = root / "timeline_normalize_errors.txt"
    if failures and not args.dry_run:
        err_path.write_text(
            "\n".join(f"{rel}\t{msg}" for rel, msg in failures) + "\n",
            encoding="utf-8",
        )
        print(f"[STATS] 失败明细已写入 {err_path.name}（共 {len(failures)} 条）")
    elif failures and args.dry_run:
        print(f"[STATS] dry-run 未写 {err_path.name}，本会写入 {len(failures)} 条失败")
    elif not failures and err_path.is_file() and not args.dry_run:
        err_path.unlink(missing_ok=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
