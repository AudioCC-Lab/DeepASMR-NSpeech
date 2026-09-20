"""
从 train.json 提取唯一 video id，用 yt-dlp 拉取 YouTube 简介时间轴并缓存到本地。

不下载音频，不写 meta，仅写 timeline.csv；已缓存的 id 默认跳过。

目录结构（默认 <root>/timeline_cache/）:
  timeline_cache/<video_id>/timeline.csv

用法:
  python collection/fetch_train_timelines.py \\
    --root data --train-json data/train.json --continue-on-error
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_code_dir = Path(__file__).resolve().parent
_repo_root = _code_dir.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from download_yt_audio_and_timeline import (  # noqa: E402
    _safe_video_id,
    _write_timeline_csv,
    parse_timeline_from_description,
    sanitize_cookies_for_youtube,
    yt_dlp_info,
)
from deepasmr_nspeech import RunSummary  # noqa: E402


def _unique_ids_from_train(train_path: Path) -> list[str]:
    data = json.loads(train_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("train.json 顶层必须是 JSON 数组")
    seen: set[str] = set()
    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("id") or "").strip()
        if vid and vid not in seen:
            seen.add(vid)
            out.append(vid)
    return sorted(out)


def _timeline_cached(cache_dir: Path, video_id: str) -> bool:
    csv_path = cache_dir / video_id / "timeline.csv"
    if not csv_path.is_file():
        return False
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 2


def _youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_and_cache_timeline(
    video_id: str,
    *,
    cache_dir: Path,
    cookies: Path | None,
    cookies_from_browser: str | None,
    skip_if_cached: bool,
) -> dict:
    if skip_if_cached and _timeline_cached(cache_dir, video_id):
        return {
            "video_id": video_id,
            "status": "skip_cached",
            "timeline_count": 0,
            "message": "timeline.csv 已存在",
        }

    url = _youtube_watch_url(video_id)
    info = yt_dlp_info(url, cookies, cookies_from_browser)
    vid = _safe_video_id(info)
    if vid != video_id:
        return {
            "video_id": video_id,
            "status": "id_mismatch",
            "timeline_count": 0,
            "message": f"yt-dlp 返回 id={vid}，与请求 {video_id} 不一致",
        }

    description = info.get("description") or ""
    timeline = parse_timeline_from_description(description)
    title = (info.get("title") or "").strip()

    per_id_dir = cache_dir / vid
    per_id_dir.mkdir(parents=True, exist_ok=True)
    _write_timeline_csv(per_id_dir / "timeline.csv", vid, timeline)

    return {
        "video_id": vid,
        "status": "ok" if timeline else "no_timeline",
        "timeline_count": len(timeline),
        "message": title,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 train.json 批量拉取 YouTube 简介 timeline 并缓存（不下载音频、不写 meta）"
    )
    ap.add_argument("--root", type=Path, required=True, help="项目根目录")
    ap.add_argument("--train-json", type=Path, required=True, help="train.json 路径")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="timeline 缓存目录（默认 <root>/timeline_cache）",
    )
    ap.add_argument("--cookies", type=Path, default=None, help="Netscape cookies.txt（会自动筛 YouTube 相关）")
    ap.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="从浏览器读取 cookies，如 edge / chrome / firefox",
    )
    ap.add_argument("--remote-ejs", type=str, default="github", help="EJS 组件来源，默认 github")
    ap.add_argument("--js-runtimes", type=str, default=None, help="如 deno")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单条失败时继续，失败 id 写入 <cache-dir>/failed_ids.txt",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="即使已有 timeline.csv 也重新拉取",
    )
    ap.add_argument(
        "--ids",
        type=str,
        default=None,
        help="仅处理逗号分隔的 video id 子集（默认从 train.json 取全部唯一 id）",
    )
    args = ap.parse_args()

    if args.cookies and args.cookies_from_browser:
        raise SystemExit("请在 --cookies 与 --cookies-from-browser 中二选一")

    train_path = args.train_json.resolve()
    if not train_path.is_file():
        raise SystemExit(f"train.json 不存在: {train_path}")

    cache_dir = (args.cache_dir or (args.root / "timeline_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed_path = cache_dir / "failed_ids.txt"

    if args.ids:
        video_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        video_ids = _unique_ids_from_train(train_path)

    if not video_ids:
        raise SystemExit("未找到任何 video id")

    print(f"[INFO] 待处理 {len(video_ids)} 个唯一 id，缓存目录: {cache_dir}")

    cookies = args.cookies
    cookies_from_browser = (args.cookies_from_browser or "").strip() or None
    skip_if_cached = not args.force

    ok = skip = no_tl = err = 0
    run_summary = RunSummary("collection.fetch_train_timelines")
    with tempfile.TemporaryDirectory(prefix="yt_cookies_") as td:
        if cookies is not None:
            cookies = sanitize_cookies_for_youtube(cookies, Path(td))

        for vid in video_ids:
            try:
                result = fetch_and_cache_timeline(
                    vid,
                    cache_dir=cache_dir,
                    cookies=cookies,
                    cookies_from_browser=cookies_from_browser,
                    skip_if_cached=skip_if_cached,
                )
                st = result["status"]
                if st == "skip_cached":
                    skip += 1
                    run_summary.add("skipped", item_id=vid, reason="already_cached")
                    print(f"[SKIP] {vid}（已缓存）")
                elif st == "ok":
                    ok += 1
                    run_summary.add("success", item_id=vid)
                    msg = result["message"]
                    print(f"[OK] {vid}  timeline={result['timeline_count']}  {msg[:60] if msg else ''}")
                elif st == "no_timeline":
                    no_tl += 1
                    run_summary.add("skipped", item_id=vid, reason="missing_timeline")
                    print(f"[NO-TIMELINE] {vid}  简介中无时间轴")
                else:
                    err += 1
                    run_summary.add("failed", item_id=vid, reason=st, detail=result["message"])
                    print(f"[WARN] {vid}  {result['message']}")
            except Exception as e:
                err += 1
                run_summary.add("failed", item_id=vid, detail=e)
                print(f"[ERR] {vid}\n  {e}")
                if args.continue_on_error:
                    with failed_path.open("a", encoding="utf-8") as f:
                        f.write(vid + "\n")
                    continue
                run_summary.finish(cache_dir / "fetch_run_summary.json")
                raise

    summary_path = cache_dir / "fetch_summary.txt"
    summary_lines = [
        f"train.json: {train_path}",
        f"唯一 id 数: {len(video_ids)}",
        f"新拉取含 timeline: {ok}",
        f"新拉取无 timeline: {no_tl}",
        f"跳过已缓存: {skip}",
        f"失败/异常: {err}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    run_summary.finish(
        cache_dir / "fetch_run_summary.json",
        metadata={"requested_video_ids": len(video_ids)},
    )
    print(f"[DONE] {summary_path.name}: ok={ok} no_timeline={no_tl} skip={skip} err={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
