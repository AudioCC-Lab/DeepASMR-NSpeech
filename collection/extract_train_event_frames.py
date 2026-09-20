"""
为 train_events.json 中每个 event 在 [start_sec, end_sec] 内随机截一帧 PNG。

输出命名: <id>_<start_sec>_<end_sec>.png（秒数与 yt_frame_extract 一致）
默认目录: <root>/fig_train/

同一 video id 共用 native-local 视频缓存，避免重复整段下载。

用法:
  python collection/extract_train_event_frames.py \\
    --root data --train-events data/train_events.json --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_code_dir = Path(__file__).resolve().parent
_repo_root = _code_dir.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from yt_frame_extract import SnippetMode, extract_one_frame, segment_output_png_name  # noqa: E402
from deepasmr_nspeech import RunSummary  # noqa: E402


def _load_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return payload["events"]
    if isinstance(payload, list):
        return payload
    raise ValueError("train_events.json 须为 {events:[...]} 或顶层数组")


def _random_capture_sec(start_sec: float, end_sec: float, rng: random.Random) -> float:
    lo, hi = float(start_sec), float(end_sec)
    if hi <= lo:
        return lo
    return rng.uniform(lo, hi)


def _youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def extract_frames_for_events(
    events: list[dict],
    *,
    fig_dir: Path,
    cache_root: Path,
    cookies: Path | None,
    cookies_browser: str | None,
    snippet_mode: SnippetMode,
    remote_ejs: str | None,
    js_runtimes: str | None,
    video_format: str,
    local_video_format: str,
    force: bool,
    continue_on_error: bool,
    rng: random.Random,
    margin_before: float,
    margin_after: float,
    force_ipv4: bool,
    socket_timeout_sec: int,
    proxy: str | None,
    yt_retries: int,
    run_summary: RunSummary,
) -> tuple[int, int, int]:
    """返回 (ok, skip, err)。"""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        vid = str(ev.get("id") or "").strip()
        if not vid:
            run_summary.add("skipped", reason="missing_video_id")
            continue
        by_id[vid].append(ev)

    ok = skip = err = 0
    failed_path = fig_dir / "failed_events.txt"

    for vid in sorted(by_id.keys()):
        id_events = by_id[vid]
        frame_cache = (
            None
            if snippet_mode != "native-local"
            else (cache_root / vid).resolve()
        )
        watch_url = _youtube_watch_url(vid)

        for ev in id_events:
            try:
                start_s = float(ev["start_sec"])
                end_s = float(ev["end_sec"])
            except (KeyError, TypeError, ValueError):
                err += 1
                run_summary.add("failed", item_id=vid, reason="invalid_time_range", detail="missing start_sec/end_sec")
                print(f"[ERR] {vid} 缺少有效 start_sec/end_sec")
                if continue_on_error:
                    continue
                run_summary.finish(fig_dir / "extract_run_summary.json")
                raise SystemExit(1)

            png_name = segment_output_png_name(vid, start_s, end_s)
            if png_name is None:
                err += 1
                run_summary.add("failed", item_id=vid, reason="invalid_time_range", detail=f"start={start_s} end={end_s}")
                print(f"[ERR] {vid} 无法生成合法文件名 start={start_s} end={end_s}")
                if continue_on_error:
                    continue
                run_summary.finish(fig_dir / "extract_run_summary.json")
                raise SystemExit(1)

            outp = fig_dir / png_name
            if outp.is_file() and outp.stat().st_size > 0 and not force:
                skip += 1
                run_summary.add("skipped", item_id=png_name, reason="already_exists")
                continue

            at_sec = _random_capture_sec(start_s, end_s, rng)
            try:
                extract_one_frame(
                    watch_url,
                    at_sec=at_sec,
                    output_png=outp,
                    cookies=cookies,
                    cookies_browser=cookies_browser,
                    margin_before=margin_before,
                    margin_after=margin_after,
                    remote_ejs=remote_ejs,
                    js_runtimes=js_runtimes,
                    video_format=video_format,
                    local_video_format=local_video_format,
                    snippet_mode=snippet_mode,
                    cache_dir=frame_cache,
                    force_kf=False,
                    ffmpeg_accurate_seek=False,
                    force_ipv4=force_ipv4,
                    socket_timeout_sec=socket_timeout_sec,
                    proxy=proxy,
                    proxy_env_override=proxy,
                    section_extra_ffmpeg_i1=True,
                    yt_retries=yt_retries,
                )
                ok += 1
                run_summary.add("success", item_id=png_name)
                print(f"[OK] {png_name}  @ {at_sec:.2f}s")
            except Exception as e:
                err += 1
                run_summary.add("failed", item_id=png_name, detail=e)
                print(f"[ERR] {png_name}  @ {at_sec:.2f}s\n  {e}")
                if continue_on_error:
                    with failed_path.open("a", encoding="utf-8") as f:
                        f.write(f"{png_name}\t{at_sec:.3f}\t{e}\n")
                    continue
                run_summary.finish(fig_dir / "extract_run_summary.json")
                raise

    return ok, skip, err


def main() -> int:
    ap = argparse.ArgumentParser(description="为 train_events 每个 event 随机截帧到 fig_train/")
    ap.add_argument("--root", type=Path, required=True, help="项目根目录")
    ap.add_argument("--train-events", type=Path, required=True, help="train_events.json 路径")
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="PNG 输出目录（默认 <root>/fig_train）",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="native-local 整段视频缓存根（默认 <fig-dir>/.video_cache）",
    )
    ap.add_argument("--cookies", type=Path, default=None)
    ap.add_argument("--cookies-from-browser", type=str, default=None)
    ap.add_argument("--remote-ejs", type=str, default="github")
    ap.add_argument("--js-runtimes", type=str, default=None)
    ap.add_argument(
        "--snippet-mode",
        choices=("ffmpeg-section", "native-local"),
        default="native-local",
        help="默认 native-local：同 id 只下载一次整段低清视频",
    )
    ap.add_argument("--video-format", type=str, default="best[height<=480]/best")
    ap.add_argument(
        "--local-video-format",
        type=str,
        default="best[height<=480]/best",
    )
    ap.add_argument("--force", action="store_true", help="已存在 PNG 也重截")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="随机取帧种子（便于复现）")
    ap.add_argument("--ids", type=str, default=None, help="仅处理逗号分隔的 video id")
    ap.add_argument("--limit", type=int, default=None, help="最多处理 event 条数（调试）")
    ap.add_argument("--margin-before", type=float, default=1.0)
    ap.add_argument("--margin-after", type=float, default=1.0)
    ap.add_argument("--force-ipv4", action="store_true")
    ap.add_argument("--socket-timeout-sec", type=int, default=120)
    ap.add_argument("--proxy", type=str, default=None)
    ap.add_argument("--yt-retries", type=int, default=3)
    args = ap.parse_args()

    if args.cookies and args.cookies_from_browser:
        raise SystemExit("请在 --cookies 与 --cookies-from-browser 中二选一")

    events_path = args.train_events.resolve()
    if not events_path.is_file():
        raise SystemExit(f"文件不存在: {events_path}")

    fig_dir = (args.fig_dir or (args.root / "fig_train")).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (args.cache_dir or (fig_dir / ".video_cache")).resolve()

    events = _load_events(events_path)
    if args.ids:
        allow = {x.strip() for x in args.ids.split(",") if x.strip()}
        events = [e for e in events if str(e.get("id") or "") in allow]
    if args.limit is not None and args.limit >= 0:
        events = events[: args.limit]

    if not events:
        raise SystemExit("没有可处理的 event")

    rng = random.Random(args.seed)
    run_summary = RunSummary("collection.extract_train_event_frames")
    cookies = args.cookies.resolve() if args.cookies else None
    cookies_browser = (args.cookies_from_browser or "").strip() or None
    proxy = args.proxy.strip() if args.proxy and args.proxy.strip() else None

    n_ids = len({str(e.get("id") or "") for e in events})
    print(f"[INFO] events={len(events)}  ids={n_ids}  out={fig_dir}  mode={args.snippet_mode}")

    ok, skip, err = extract_frames_for_events(
        events,
        fig_dir=fig_dir,
        cache_root=cache_root,
        cookies=cookies,
        cookies_browser=cookies_browser,
        snippet_mode=args.snippet_mode,
        remote_ejs=(args.remote_ejs or None),
        js_runtimes=args.js_runtimes,
        video_format=args.video_format,
        local_video_format=args.local_video_format,
        force=args.force,
        continue_on_error=args.continue_on_error,
        rng=rng,
        margin_before=args.margin_before,
        margin_after=args.margin_after,
        force_ipv4=args.force_ipv4,
        socket_timeout_sec=args.socket_timeout_sec,
        proxy=proxy,
        yt_retries=max(1, args.yt_retries),
        run_summary=run_summary,
    )

    summary = fig_dir / "extract_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"train_events: {events_path}",
                f"processed: {len(events)}",
                f"ok: {ok}",
                f"skip: {skip}",
                f"err: {err}",
                f"seed: {args.seed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_summary.finish(
        fig_dir / "extract_run_summary.json",
        metadata={"requested_events": len(events), "seed": args.seed},
    )
    print(f"[DONE] ok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
