from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepasmr_nspeech import RunSummary  # noqa: E402


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"命令失败: {' '.join(cmd)}\n{msg}")
    return p.stdout


@dataclass(frozen=True)
class VideoUrlItem:
    url: str
    video_id: str | None
    title: str | None


def _iter_channel_video_urls(
    channel_videos_url: str,
    *,
    cookies: Path | None,
    cookies_from_browser: str | None,
    limit: int,
    include_shorts: bool,
) -> list[VideoUrlItem]:
    """
    使用 yt-dlp 从频道 /videos 页面获取视频列表（flat playlist），避免自己抓 HTML。
    """
    cmd = [
        "yt-dlp",
        "-J",
        "--flat-playlist",
        "--no-warnings",
        "--playlist-end",
        str(max(1, limit)),
        channel_videos_url,
    ]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    if cookies is not None:
        cmd[1:1] = ["--cookies", str(cookies)]

    data = json.loads(_run(cmd))
    entries = data.get("entries") or []

    out: list[VideoUrlItem] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        vid = (e.get("id") or "").strip() or None
        title = e.get("title")
        u = e.get("url") or ""
        if isinstance(u, str):
            u = u.strip()
        else:
            u = ""

        # yt-dlp flat-playlist 的 url 可能是 id 或完整链接；尽量转换成 watch?v= 形式
        if u.startswith("http://") or u.startswith("https://"):
            url = u
        elif vid:
            url = f"https://www.youtube.com/watch?v={vid}"
        elif u:
            # 兜底：某些情况下 url 字段直接就是 id
            url = f"https://www.youtube.com/watch?v={u}"
        else:
            continue

        if not include_shorts and "/shorts/" in url:
            continue

        out.append(VideoUrlItem(url=url, video_id=vid, title=title))

    return out


def _write_urls_file(items: Iterable[VideoUrlItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for it in items:
        lines.append(it.url)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _call_single_download(
    downloader_py: Path,
    *,
    url: str,
    root: Path,
    audio_dir: Path | None,
    cookies: Path | None,
    cookies_from_browser: str | None,
    audio_format: str,
    audio_quality: str,
    outtmpl: str,
    remote_ejs: str | None,
    js_runtimes: str | None,
    require_timeline: bool,
    title_keywords: str | None,
    write_ingest_summary: bool,
) -> None:
    cmd = [
        sys.executable,
        str(downloader_py),
        "--url",
        url,
        "--root",
        str(root),
        "--audio-format",
        audio_format,
        "--audio-quality",
        audio_quality,
        "--outtmpl",
        outtmpl,
    ]
    if audio_dir is not None:
        cmd += ["--audio-dir", str(audio_dir)]
    if cookies is not None:
        cmd += ["--cookies", str(cookies)]
    if cookies_from_browser is not None:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if remote_ejs is not None:
        cmd += ["--remote-ejs", remote_ejs]
    if js_runtimes is not None:
        cmd += ["--js-runtimes", js_runtimes]
    if require_timeline:
        cmd.append("--require-timeline")
    if title_keywords is not None and str(title_keywords).strip():
        cmd += ["--title-keywords", str(title_keywords).strip()]
    if write_ingest_summary:
        cmd.append("--write-ingest-summary")

    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="输入 YouTube 频道 /videos 页面，获取视频 URL，并可选择批量下载"
    )
    ap.add_argument("--channel-videos-url", type=str, required=True, help="例如 https://www.youtube.com/@xx/videos")
    ap.add_argument("--cookies", type=Path, default=None, help="Netscape cookies.txt（可选；需要登录态时填）")
    ap.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="直接从浏览器读取 cookies（可选）：例如 edge / chrome / firefox。与 --cookies 二选一",
    )
    ap.add_argument("--limit", type=int, default=9999, help="仅拉取前 N 条视频 URL（默认 9999）")
    ap.add_argument("--include-shorts", action="store_true", help="包含 shorts（默认不包含）")
    ap.add_argument("--urls-out", type=Path, default=None, help="把解析到的 URL 写入该文件（可选）")
    ap.add_argument(
        "--download",
        type=str,
        choices=["all", "first", "none"],
        default="all",
        help="下载模式：all=全部下载（默认）；first=仅下载第一条；none=只抓 URL 不下载",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="批量下载时，遇到单条失败则跳过继续（失败 URL 会写入 <root>\\failed_urls.txt）",
    )

    # 透传给 download_yt_audio_and_timeline.py 的参数（命名与其它脚本一致：--root / --audio-dir）
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="元数据根目录（写 export.csv、meta.json 等；与 download_yt_audio_and_timeline --root 相同）",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（可选，不填则 <root>\\audio）",
    )
    ap.add_argument("--audio-format", type=str, default="wav", help="音频格式：wav/m4a/mp3/flac...（默认 wav）")
    ap.add_argument("--audio-quality", type=str, default="0", help="音频质量（yt-dlp 透传给 ffmpeg），默认 0（最佳）")
    ap.add_argument("--outtmpl", type=str, default="%(id)s.%(ext)s", help="输出文件名模板（相对 --audio-dir）")
    ap.add_argument(
        "--remote-ejs",
        type=str,
        default="github",
        help="EJS 组件来源（用于解决 YouTube n challenge）。默认 github；传空字符串可关闭",
    )
    ap.add_argument("--js-runtimes", type=str, default=None, help="指定 JS 运行时（例如 deno 或 node）")
    ap.add_argument(
        "--require-timeline",
        action="store_true",
        help="透传：简介中无时间轴则不下载、不写库",
    )
    ap.add_argument(
        "--title-keywords",
        type=str,
        default=None,
        metavar="KW,KW2",
        help="透传：逗号分隔（每项可含空格），标题须含其一子串（忽略大小写）才下载",
    )
    ap.add_argument(
        "--write-ingest-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="透传：写入数据清单统计；默认开启",
    )

    args = ap.parse_args()
    if args.cookies and args.cookies_from_browser:
        raise SystemExit("请在 --cookies 与 --cookies-from-browser 中二选一")

    items = _iter_channel_video_urls(
        args.channel_videos_url,
        cookies=args.cookies,
        cookies_from_browser=(args.cookies_from_browser or "").strip() or None,
        limit=args.limit,
        include_shorts=args.include_shorts,
    )
    if not items:
        raise SystemExit("未解析到任何视频 URL（检查频道地址、网络、cookies 或 yt-dlp 版本）")

    if args.urls_out is not None:
        _write_urls_file(items, args.urls_out)
        print(f"[OK] 已写入 URL 列表: {args.urls_out}")

    downloader_py = Path(__file__).with_name("download_yt_audio_and_timeline.py")
    if not downloader_py.exists():
        raise SystemExit(f"未找到下载脚本: {downloader_py}")

    if args.download == "none":
        print("[INFO] download=none，仅抓取 URL，不下载。")
        return 0

    urls = [it.url for it in items]
    if args.download == "first":
        urls = urls[:1]
        print(f"[INFO] download=first，仅下载 1 条: {urls[0]}")
    else:
        print(f"[INFO] download=all，准备下载 {len(urls)} 条")

    run_summary = RunSummary("collection.channel_download")
    for u in urls:
        cmd_args = dict(
            downloader_py=downloader_py,
            url=u,
            root=args.root,
            audio_dir=args.audio_dir,
            cookies=args.cookies,
            cookies_from_browser=(args.cookies_from_browser or "").strip() or None,
            audio_format=args.audio_format,
            audio_quality=args.audio_quality,
            outtmpl=args.outtmpl,
            remote_ejs=(args.remote_ejs or None),
            js_runtimes=args.js_runtimes,
            require_timeline=bool(args.require_timeline),
            title_keywords=args.title_keywords,
            write_ingest_summary=bool(args.write_ingest_summary),
        )
        try:
            _call_single_download(**cmd_args)
            child_summary_path = args.root / "download_run_summary.json"
            child = {}
            if child_summary_path.is_file():
                try:
                    child = json.loads(child_summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    child = {}
            counts = child.get("counts") or {}
            if int(child.get("failed") or 0) > 0:
                failures = child.get("failures") or []
                detail = failures[0].get("detail") if failures else "child downloader failed"
                run_summary.add("failed", item_id=u, detail=detail)
            elif int(counts.get("skipped") or 0) > 0:
                reasons = child.get("failure_reasons") or {}
                reason = next(iter(reasons), "filtered_or_complete")
                run_summary.add("skipped", item_id=u, reason=reason)
            else:
                run_summary.add("success", item_id=u)
        except subprocess.CalledProcessError as e:
            print(f"[ERR] {u}")
            print(f"  exit_code={e.returncode}")
            run_summary.add("failed", item_id=u, detail=f"downloader exit code {e.returncode}")
            if args.continue_on_error:
                failed_urls_path = args.root / "failed_urls.txt"
                failed_urls_path.parent.mkdir(parents=True, exist_ok=True)
                with failed_urls_path.open("a", encoding="utf-8") as f:
                    f.write(u + "\n")
                continue
            run_summary.finish(args.root / "channel_download_run_summary.json")
            raise

    run_summary.finish(
        args.root / "channel_download_run_summary.json",
        metadata={"channel": args.channel_videos_url, "requested_urls": len(urls)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

