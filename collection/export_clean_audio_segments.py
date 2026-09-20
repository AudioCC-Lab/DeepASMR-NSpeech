"""
按 segment_label_parse.json 的 clean_start/clean_end 导出 WAV 切片。

输出：<segments-dir>/<id>_<cleanStart>_<cleanEnd>/audio.wav + info.json
info.json 仅含：id, clean_start_clean_end, title, label, seconds。
title：优先用 segment 内已有 title，否则从 <root>/meta.json（可用 --meta）按视频 id 取标题。

默认增量：目标目录下已有非空 audio.wav 且存在 info.json 则跳过（可用 --overwrite 强制重做）。

建议流水线：clean_audio_cut → 本脚本 → yt_frame_extract.py --auto-segments → run_label_parsing.py --from-segments。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_code_dir = Path(__file__).resolve().parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from segment_layout import AUDIO_WAV, INFO_JSON  # noqa: E402
from yt_frame_extract import safe_fname_num  # noqa: E402


DEFAULT_SEGMENT_JSON_NAME = "segment_label_parse.json"


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


def run_ffmpeg(ff: List[str]) -> None:
    p = subprocess.run(ff, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 失败 ({p.returncode})\n{err}")


def segment_folder_name(video_id: str, clean_start: float, clean_end: float) -> str:
    return f"{video_id}_{safe_fname_num(clean_start)}_{safe_fname_num(clean_end)}"


def clean_start_clean_end_field(clean_start: float, clean_end: float) -> str:
    return f"{safe_fname_num(clean_start)}_{safe_fname_num(clean_end)}"


def parse_clean_window(seg: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    cs, ce = seg.get("clean_start"), seg.get("clean_end")
    if cs is None or ce is None:
        return None
    try:
        a, b = float(cs), float(ce)
    except (TypeError, ValueError):
        return None
    if b <= a:
        return None
    return a, b


def load_segments(json_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("invalid json root")
    vid = (payload.get("video_id") or "").strip()
    raw = payload.get("segments")
    if not isinstance(raw, list):
        return vid, []
    segs = [x for x in raw if isinstance(x, dict)]
    return vid, segs


def _read_meta_titles(meta_path: Path) -> Dict[str, str]:
    """与 label_parsing / 下载流水线一致：meta.json 内各视频的 title。"""
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                tid = (v.get("id") or k or "").strip()
                title = (v.get("title") or "").strip()
                if tid:
                    out[tid] = title
    return out


def resolve_segment_title(
    seg: Dict[str, Any],
    video_id: str,
    meta_titles: Dict[str, str],
) -> Optional[str]:
    """段内 title 优先，否则 meta 中该 id 的视频标题。"""
    raw = seg.get("title")
    if isinstance(raw, str):
        t = raw.strip()
        if t:
            return t
    elif raw is not None:
        s = str(raw).strip()
        if s:
            return s
    mt = (meta_titles.get(video_id) or "").strip()
    return mt if mt else None


def export_one_clip(
    *,
    src_audio: Path,
    dest_dir: Path,
    video_id: str,
    clean_start: float,
    clean_end: float,
    title: Optional[str],
    label: Optional[str],
    overwrite: bool,
    dry_run: bool,
) -> str:
    """返回 skipped | written | dry-run"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav_path = dest_dir / AUDIO_WAV
    info_path = dest_dir / INFO_JSON

    key_field = clean_start_clean_end_field(clean_start, clean_end)
    seconds = float(clean_end - clean_start)

    info_payload = {
        "id": video_id,
        "clean_start_clean_end": key_field,
        "title": title if title is not None else None,
        "label": label if label is not None else None,
        "seconds": seconds,
    }

    if (
        not overwrite
        and wav_path.is_file()
        and wav_path.stat().st_size > 0
        and info_path.is_file()
    ):
        return "skipped"

    if dry_run:
        return "dry-run"

    duration = seconds
    ff = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_audio.resolve()),
        "-ss",
        str(clean_start),
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(wav_path.resolve()),
    ]
    run_ffmpeg(ff)

    with info_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(info_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return "written"


def process_one_id(
    video_id: str,
    id_dir: Path,
    *,
    segments_root: Path,
    segment_json_name: str,
    meta_titles: Dict[str, str],
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int, int, Optional[str]]:
    """returns written, skipped, dry_run_count, error_msg"""
    json_path = id_dir / segment_json_name
    if not json_path.is_file():
        return 0, 0, 0, "no segment_label_parse.json"

    try:
        vid_json, segs = load_segments(json_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return 0, 0, 0, str(e)

    vid = (vid_json or video_id).strip() or video_id

    src = find_audio_in_id_dir(id_dir, vid)
    if src is None:
        return 0, 0, 0, "no audio file"

    written = skipped = dry_c = 0
    seen_dirs: set[str] = set()

    for seg in segs:
        win = parse_clean_window(seg)
        if win is None:
            continue
        cs, ce = win
        folder = segment_folder_name(vid, cs, ce)
        if folder in seen_dirs:
            continue
        seen_dirs.add(folder)

        dest_dir = segments_root / folder
        title = resolve_segment_title(seg, vid, meta_titles)

        lab = seg.get("label")
        if isinstance(lab, str):
            lab = lab.strip() or None
        elif lab is not None:
            lab = str(lab)

        try:
            status = export_one_clip(
                src_audio=src,
                dest_dir=dest_dir,
                video_id=vid,
                clean_start=cs,
                clean_end=ce,
                title=title,
                label=lab,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        except Exception as e:
            return written, skipped, dry_c, f"clip {folder}: {e}"

        if status == "written":
            written += 1
        elif status == "skipped":
            skipped += 1
        elif status == "dry-run":
            dry_c += 1

    if not seen_dirs:
        return 0, 0, 0, "no segments with valid clean_start/clean_end"

    return written, skipped, dry_c, None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="按 clean 窗导出 WAV 到 <segments-dir>/<id>_<cs>_<ce>/（增量默认跳过已存在）",
    )
    ap.add_argument("--root", type=Path, default=_code_dir.parent, help="项目根目录（默认 code 上级）")
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="meta.json 路径（默认：<root>/meta.json）；用于填充 info.json 的 title",
    )
    ap.add_argument(
        "--audio-dir",
        "--audio-folder",
        type=Path,
        default=None,
        dest="audio_dir",
        help="各视频 id 子目录的父路径（默认 <root>/audio）",
    )
    ap.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="切片输出根目录（默认 <root>/segments）",
    )
    ap.add_argument("--ids", type=str, default="", help="仅处理这些 id，逗号分隔")
    ap.add_argument("--id-dir", type=Path, default=None, help="只处理单个 id 目录")
    ap.add_argument(
        "--segment-json-name",
        type=str,
        default=DEFAULT_SEGMENT_JSON_NAME,
        help="JSON 文件名",
    )
    ap.add_argument("--overwrite", action="store_true", help="已存在 wav/info 也重新切割")
    ap.add_argument("--dry-run", action="store_true", help="只打印将处理的目录，不调用 ffmpeg")
    args = ap.parse_args()

    root = args.root.resolve()
    audio_dir = (args.audio_dir or (root / "audio")).resolve()
    segments_root = (args.segments_dir or (root / "segments")).resolve()
    meta_path = (args.meta or (root / "meta.json")).resolve()
    meta_titles = _read_meta_titles(meta_path)
    if not meta_path.is_file():
        print(f"[WARN] 未找到 meta.json，title 仅来自 segment（可为 null）: {meta_path}", file=sys.stderr)

    id_filter: Optional[set[str]] = None
    if (args.ids or "").strip():
        id_filter = {x.strip() for x in args.ids.split(",") if x.strip()}

    if args.id_dir is not None:
        id_path = args.id_dir.resolve()
        if not id_path.is_dir():
            print(f"[ERR] --id-dir 不是目录: {id_path}", file=sys.stderr)
            return 1
        subdirs = [id_path]
    else:
        if not audio_dir.is_dir():
            print(f"[ERR] audio 目录不存在: {audio_dir}", file=sys.stderr)
            return 1
        subdirs = sorted([p for p in audio_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())

    segments_root.mkdir(parents=True, exist_ok=True)

    tot_w = tot_s = tot_d = 0
    errs: List[str] = []
    for d in subdirs:
        vid = d.name
        if id_filter is not None and vid not in id_filter:
            continue
        w, s, dr, err = process_one_id(
            vid,
            d,
            segments_root=segments_root,
            segment_json_name=args.segment_json_name,
            meta_titles=meta_titles,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        tot_w += w
        tot_s += s
        tot_d += dr
        if err:
            errs.append(f"{vid}: {err}")
            continue
        tag = "[DRY-RUN]" if args.dry_run else "[OK]"
        print(f"{tag} {vid} written={w} skipped={s} dry={dr} -> {segments_root}")

    print(f"完成: 写入切片={tot_w}, 跳过={tot_s}, dry-run={tot_d}, 失败 id={len(errs)}")
    for e in errs[:30]:
        print(f"  {e}", file=sys.stderr)
    if len(errs) > 30:
        print(f"  ... 另有 {len(errs)-30} 条", file=sys.stderr)

    return 0 if not errs else 2


if __name__ == "__main__":
    raise SystemExit(main())
