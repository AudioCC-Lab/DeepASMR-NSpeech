"""
包装 clean_audio_cut：按 id 处理 segment_label_parse.json 与声学裁窗。

逻辑：
  1. 若无 segment_label_parse.json：从 timeline.csv 生成占位 JSON（键名与 LLM 产物一致，
     LLM 字段为 null/空串；clean_start / clean_end 置 null），再调用 clean_audio_cut 写入裁窗。
  2. 若已有 JSON：若任一段缺少 clean_start 或 clean_end（含值为 null），则调用 clean_audio_cut；
     否则跳过。
  3. 其余参数透传给 clean_audio_cut.py（ruptures / --window-sec、--penalty、--audio-file 等）。

示例：
  python clean_audio_cut_wrap.py --audio-folder "D:/youtube_asmr/audio"
  python clean_audio_cut_wrap.py --audio-folder "D:/youtube_asmr/audio" --ids "xc5otEPVONo"
  python clean_audio_cut_wrap.py --root "D:/youtube_asmr" --window-sec 4.0 --penalty 20
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

from clean_audio_cut import read_timeline_segments_with_rows  # noqa: E402
from yt_frame_extract import segment_output_png_name  # noqa: E402

SEGMENT_JSON_NAME_DEFAULT = "segment_label_parse.json"


def _read_meta_title(meta_path: Path, video_id: str) -> str:
    if not meta_path.is_file():
        return ""
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    for k, v in raw.items():
        if isinstance(v, dict):
            tid = (v.get("id") or k or "").strip()
            title = (v.get("title") or "").strip()
            if tid == video_id and title:
                return title
            if tid == video_id:
                return title
    return ""


def _keyframe_relative(video_id: str, start: float, end: float) -> Optional[str]:
    name = segment_output_png_name(video_id, start, end)
    if not name:
        return None
    return str(Path("frames") / name)


def empty_segment_dict(
    *,
    video_id: str,
    start: float,
    end: float,
    seg_title: str,
    video_title: str,
) -> Dict[str, Any]:
    """与现有 segment_label_parse 对齐的占位段；LLM 后续补全。"""
    kf = _keyframe_relative(video_id, start, end)
    label = (seg_title or "").strip() or None
    vt = (video_title or "").strip() or None
    return {
        "video_id": video_id,
        "start_sec": float(start),
        "end_sec": float(end),
        "title": vt,
        "label": label,
        "keyframe_relative": kf,
        "verb": None,
        "verb_source": None,
        "subject": None,
        "subject_source": None,
        "object": None,
        "object_source": None,
        "device": None,
        "device_source": None,
        "visual_role": None,
        "notes": None,
        "clean_start": None,
        "clean_end": None,
    }


def build_stub_payload(
    video_id: str,
    id_dir: Path,
    *,
    segment_json_name: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    tl = id_dir / "timeline.csv"
    if not tl.is_file():
        return None, "no timeline.csv"
    try:
        _, _, segs = read_timeline_segments_with_rows(tl, video_id)
    except Exception as e:
        return None, str(e)
    if not segs:
        return None, "no valid timeline rows"

    meta_title = _read_meta_title(id_dir / "meta.json", video_id)
    segments = [
        empty_segment_dict(
            video_id=vid,
            start=st,
            end=en,
            seg_title=seg_title,
            video_title=meta_title,
        )
        for vid, st, en, seg_title in segs
    ]
    payload: Dict[str, Any] = {
        "video_id": video_id,
        "source_timeline": "timeline.csv",
        "source_meta_title": "meta.json",
        "llm_model": None,
        "segment_count": len(segments),
        "segments": segments,
    }
    return payload, None


def segment_needs_clean(seg: Any) -> bool:
    if not isinstance(seg, dict):
        return True
    if "clean_start" not in seg or "clean_end" not in seg:
        return True
    if seg.get("clean_start") is None or seg.get("clean_end") is None:
        return True
    return False


def payload_needs_clean(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return True
    segs = payload.get("segments")
    if not isinstance(segs, list) or not segs:
        return True
    return any(segment_needs_clean(s) for s in segs)


def run_clean_audio_cut(
    id_dir: Path,
    passthrough: List[str],
    *,
    dry_run: bool,
) -> int:
    if dry_run:
        print(f"  [DRY-RUN] would run: clean_audio_cut.py --id-dir {id_dir} {' '.join(passthrough)}")
        return 0
    script = _code_dir / "clean_audio_cut.py"
    cmd = [sys.executable, str(script), "--id-dir", str(id_dir), *passthrough]
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def process_one_id_folder(
    id_dir: Path,
    *,
    segment_json_name: str,
    dry_run: bool,
    passthrough: List[str],
) -> Tuple[str, Optional[str]]:
    """
    Returns (status, error_message).
    status: skipped | stub_created | cleaned | stub_and_cleaned | error
    """
    video_id = id_dir.name
    json_path = id_dir / segment_json_name

    if not json_path.is_file():
        payload, err = build_stub_payload(video_id, id_dir, segment_json_name=segment_json_name)
        if err:
            return "error", err
        if dry_run:
            print(f"[DRY-RUN] would create {json_path} ({payload['segment_count']} segments)")
        else:
            with json_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"[STUB] {video_id} -> {json_path}")
        rc = run_clean_audio_cut(id_dir.resolve(), passthrough, dry_run=dry_run)
        if rc != 0:
            return "error", f"clean_audio_cut exit {rc}"
        return "stub_and_cleaned", None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return "error", str(e)

    if not payload_needs_clean(payload):
        return "skipped", None

    print(f"[CLEAN] {video_id} missing clean_start/clean_end -> clean_audio_cut")
    rc = run_clean_audio_cut(id_dir.resolve(), passthrough, dry_run=dry_run)
    if rc != 0:
        return "error", f"clean_audio_cut exit {rc}"
    return "cleaned", None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ensure segment_label_parse.json exists with full keys; run clean_audio_cut when clean_* missing.",
    )
    ap.add_argument("--root", type=Path, default=_code_dir.parent, help="项目根目录")
    ap.add_argument(
        "--audio-dir",
        "--audio-folder",
        type=Path,
        default=None,
        dest="audio_dir",
        help="audio 根目录：遍历子目录全量处理（默认 <root>/audio）",
    )
    ap.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        dest="audio_path_alias",
        help="同 --audio-dir",
    )
    ap.add_argument("--ids", type=str, default="", help="仅处理这些 id，逗号分隔")
    ap.add_argument("--dry-run", action="store_true", help="不写入、不调用 clean_audio_cut")
    ap.add_argument(
        "--segment-json-name",
        type=str,
        default=SEGMENT_JSON_NAME_DEFAULT,
        help="JSON 文件名，默认 segment_label_parse.json",
    )
    args, passthrough = ap.parse_known_args()

    root = args.root.resolve()
    audio_root = (args.audio_path_alias or args.audio_dir or (root / "audio")).resolve()
    if not audio_root.is_dir():
        print(f"[ERR] audio 目录不存在: {audio_root}", file=sys.stderr)
        return 1

    id_filter: Optional[set[str]] = None
    if (args.ids or "").strip():
        id_filter = {x.strip() for x in args.ids.split(",") if x.strip()}

    subdirs = sorted([p for p in audio_root.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())

    n_skip = n_stub = n_clean = n_err = 0
    errs: List[str] = []

    for d in subdirs:
        vid = d.name
        if id_filter is not None and vid not in id_filter:
            continue
        status, err = process_one_id_folder(
            d,
            segment_json_name=args.segment_json_name,
            dry_run=bool(args.dry_run),
            passthrough=passthrough,
        )
        if status == "skipped":
            n_skip += 1
            print(f"[SKIP] {vid}")
        elif status == "stub_and_cleaned":
            n_stub += 1
        elif status == "cleaned":
            n_clean += 1
        elif status == "error":
            n_err += 1
            msg = f"{vid}: {err}"
            errs.append(msg)
            print(f"[ERR] {msg}", file=sys.stderr)

    print(
        f"完成: 跳过(已有裁窗)={n_skip}, 新建占位并裁窗={n_stub}, 仅补裁窗={n_clean}, 失败={n_err}"
    )
    for e in errs[:30]:
        print(f"  {e}", file=sys.stderr)

    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
