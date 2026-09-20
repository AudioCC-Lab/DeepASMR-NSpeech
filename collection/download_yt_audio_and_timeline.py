from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from timeline_parse import extract_timestamp_prefix, parse_seconds_from_timestamp_token

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepasmr_nspeech import RunSummary  # noqa: E402

YOUTUBE_WATCH_ID_RE = re.compile(r"[?&]v=([^&]+)")


@dataclass(frozen=True)
class TimelineItem:
    time_str: str
    seconds: int
    title: str
    raw_line: str


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"命令失败: {' '.join(cmd)}\n{msg}")
    return p.stdout


def _parse_csv_keywords_arg(s: str | None) -> list[str] | None:
    """逗号分隔关键词 → 小写列表；无有效项则 None（表示不按关键词过滤）。"""
    if s is None or not str(s).strip():
        return None
    parts = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    return parts or None


def _title_matches_any_keyword(title: str, keywords_lc: list[str]) -> bool:
    t = (title or "").lower()
    return any(kw in t for kw in keywords_lc)


def parse_timeline_from_description(description: str) -> list[TimelineItem]:
    if not description:
        return []

    items: list[TimelineItem] = []
    for line in description.splitlines():
        got = extract_timestamp_prefix(line)
        if got is None:
            continue
        t, title = got
        sec = parse_seconds_from_timestamp_token(t)
        if sec is None:
            continue
        items.append(TimelineItem(time_str=t, seconds=sec, title=title, raw_line=line.rstrip("\n")))

    # 去重：同一秒多条时保留第一条
    seen: set[int] = set()
    dedup: list[TimelineItem] = []
    for it in items:
        if it.seconds in seen:
            continue
        seen.add(it.seconds)
        dedup.append(it)
    return dedup


def yt_dlp_info(url: str, cookies: Path | None, cookies_from_browser: str | None) -> dict:
    cmd = ["yt-dlp", "-J", "--no-playlist", url]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    if cookies is not None:
        cmd[1:1] = ["--cookies", str(cookies)]
    out = _run(cmd)
    return json.loads(out)


def ytdlp_prepend_standard_options(
    cmd: list[str],
    *,
    cookies: Path | None,
    cookies_from_browser: str | None,
    remote_ejs: str | None,
    js_runtimes: str | None,
    force_ipv4: bool = False,
    socket_timeout_sec: int | None = None,
    proxy: str | None = None,
) -> None:
    """
    在「yt-dlp」字样之后插入认证与可选网络参数，插入顺序与同文件里的 yt_dlp_download_audio 原逻辑一致
    （cookies 紧靠 yt-dlp，再往外是浏览器 cookies / js-runtimes / remote-components；再往外是 ipv4 / 超时 / 代理）。
    """
    if not cmd or cmd[0] != "yt-dlp":
        raise ValueError("cmd must be a yt-dlp argv list")

    if remote_ejs:
        cmd[1:1] = ["--remote-components", f"ejs:{remote_ejs}"]
    if js_runtimes:
        cmd[1:1] = ["--js-runtimes", js_runtimes]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    if cookies is not None:
        cmd[1:1] = ["--cookies", str(cookies)]
    if proxy is not None:
        cmd[1:1] = ["--proxy", str(proxy)]
    if socket_timeout_sec is not None:
        cmd[1:1] = ["--socket-timeout", str(socket_timeout_sec)]
    if force_ipv4:
        cmd[1:1] = ["--force-ipv4"]


def yt_dlp_download_audio(
    url: str,
    cookies: Path | None,
    cookies_from_browser: str | None,
    audio_dir: Path,
    audio_format: str,
    audio_quality: str,
    outtmpl: str,
    *,
    remote_ejs: str | None = None,
    js_runtimes: str | None = None,
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--continue",
        "--no-overwrites",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        audio_format,
        "--audio-quality",
        audio_quality,
        "-o",
        str(audio_dir / outtmpl),
        url,
    ]
    ytdlp_prepend_standard_options(
        cmd,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        remote_ejs=remote_ejs,
        js_runtimes=js_runtimes,
    )
    _run(cmd)


def yt_dlp_download_video_for_local_snippet(
    url: str,
    cookies: Path | None,
    cookies_from_browser: str | None,
    output_template: str | Path,
    video_format: str,
    *,
    remote_ejs: str | None = None,
    js_runtimes: str | None = None,
    force_ipv4: bool = False,
    socket_timeout_sec: int | None = 120,
    proxy: str | None = None,
    extractor_args: str | None = None,
) -> None:
    """
    与 yt_dlp_download_audio 同类：仅用 yt-dlp 自身（及必要时对本地分片合并）拉流，不使用 --download-sections。
    通常整段下载后再用 ffmpeg 对本地文件 seek 截帧，可避免 ffmpeg 直接打开 googlevideo URL。
    """
    ot = Path(output_template)
    ot.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        "yt-dlp",
        "--continue",
        "--no-overwrites",
        "--no-playlist",
        "-f",
        video_format,
        "-o",
        str(output_template),
    ]
    ytdlp_prepend_standard_options(
        cmd,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        remote_ejs=remote_ejs,
        js_runtimes=js_runtimes,
        force_ipv4=force_ipv4,
        socket_timeout_sec=socket_timeout_sec,
        proxy=proxy,
    )
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    cmd.append(url)
    _run(cmd)


_YOUTUBE_COOKIE_DOMAIN_KEYWORDS = (
    "youtube.com",
    "google.com",
    "accounts.google.com",
    "googlevideo.com",
    "ytimg.com",
)


def filter_youtube_cookie_lines(text: str) -> tuple[list[str], int, int]:
    """
    从 Netscape cookies 文本中筛出 YouTube/Google 相关行。
    返回 (输出行列表, 保留的数据行数, 跳过的非法行数)。
    """
    out_lines: list[str] = []
    kept_data = 0
    skipped_invalid = 0

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        cols = line.split("\t")
        if len(cols) != 7:
            cols = line.split()
        if len(cols) != 7:
            skipped_invalid += 1
            continue

        domain = cols[0].strip()
        include_sub = cols[1].strip().upper()

        d = domain.lstrip(".").lower()
        if not any(k in d for k in _YOUTUBE_COOKIE_DOMAIN_KEYWORDS):
            continue

        if domain.startswith(".") and include_sub == "FALSE":
            cols[1] = "TRUE"

        out_lines.append("\t".join(cols))
        kept_data += 1

    return out_lines, kept_data, skipped_invalid


def write_youtube_cookies_filtered(src: Path, dst: Path) -> tuple[int, int]:
    """将 src 中 YouTube/Google 相关 cookies 写入 dst。返回 (保留的数据行数, 跳过的非法行数)。"""
    text = src.read_text(encoding="utf-8", errors="replace")
    out_lines, kept, skipped = filter_youtube_cookie_lines(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return kept, skipped


def sanitize_cookies_for_youtube(cookies_path: Path, tmp_dir: Path) -> Path:
    """
    1) 只保留 YouTube/Google 相关域名的 cookies，避免 all_cookies 里其它站点的异常行导致 yt-dlp 读取失败
    2) 修正少数导出器的格式问题：domain 以 '.' 开头但 includeSubdomains=FALSE -> TRUE
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "cookies_youtube_sanitized.txt"
    write_youtube_cookies_filtered(cookies_path, out_path)
    return out_path


def _iter_urls(url: str | None, url_file: Path | None) -> Iterable[str]:
    if url:
        yield url.strip()
        return
    if not url_file:
        return
    text = url_file.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield s


def _extract_id_from_url(url: str) -> str | None:
    m = YOUTUBE_WATCH_ID_RE.search(url or "")
    if not m:
        return None
    vid = (m.group(1) or "").strip()
    return vid or None


def _safe_video_id(info: dict) -> str:
    vid = str(info.get("id") or "").strip()
    if vid:
        return vid
    # fallback：极少数情况下没有 id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _meta_object_from_info(info: dict) -> dict:
    vid = _safe_video_id(info)
    title = info.get("title")
    uploader = info.get("uploader") or info.get("channel") or info.get("uploader_id")
    upload_date = info.get("upload_date")  # YYYYMMDD
    description = info.get("description") or ""

    return {
        "id": vid,
        "webpage_url": info.get("webpage_url") or info.get("original_url"),
        "title": title,
        "uploader": uploader,
        "upload_date": upload_date,
        "duration": info.get("duration"),
        "description": description,
    }

def _load_meta_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}

    # 兼容可能出现的几种历史格式
    if isinstance(data, dict):
        # 1) {id: {...}} 或 2) 单个 meta object（含 id）
        if "id" in data and isinstance(data.get("id"), str):
            return {data["id"]: data}
        out: dict[str, dict] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, dict):
                out[k] = v
        return out
    if isinstance(data, list):
        out = {}
        for v in data:
            if isinstance(v, dict) and isinstance(v.get("id"), str):
                out[v["id"]] = v
        return out
    return {}


def _update_meta_index(meta_index_path: Path, meta_obj: dict) -> None:
    vid = str(meta_obj.get("id") or "").strip()
    if not vid:
        return
    meta_index_path.parent.mkdir(parents=True, exist_ok=True)
    idx = _load_meta_index(meta_index_path)
    idx[vid] = meta_obj
    meta_index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_timeline_csv(path: Path, video_id: str, timeline: list[TimelineItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "time", "seconds", "segment_title"])
        for it in timeline:
            w.writerow([video_id, it.time_str, it.seconds, it.title])


def _find_downloaded_audio(audio_dir: Path, video_id: str, audio_format: str) -> Path | None:
    # 尽量匹配 {id}.{ext}；若 outtmpl 改过导致文件名不同，兜底用 {id}.* 取最新的一个
    exact = audio_dir / f"{video_id}.{audio_format}"
    if exact.exists():
        return exact
    candidates = sorted(audio_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _relpath_or_name(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def append_export_csv(
    export_csv_path: Path,
    *,
    webpage_url: str | None,
    uploader: str | None,
    audio_filename: str,
    title: str | None,
    timeline: list[TimelineItem],
) -> None:
    export_csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not export_csv_path.exists()

    def _timeline_one_line(items: list[TimelineItem]) -> str:
        if not items:
            return ""
        parts: list[str] = []
        for it in items:
            seg = it.time_str
            if it.title:
                seg = f"{seg} {it.title}"
            parts.append(seg.strip())
        return " | ".join(parts)

    with export_csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["webpage_url", "uploader", "audio_filename", "title", "timeline"])

        w.writerow([webpage_url or "", uploader or "", audio_filename, title or "", _timeline_one_line(timeline)])


def _load_export_csv_rows(export_csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not export_csv_path.exists():
        return (["webpage_url", "uploader", "audio_filename", "title", "timeline"], [])
    with export_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or ["webpage_url", "uploader", "audio_filename", "title", "timeline"]
        rows = []
        for row in r:
            if isinstance(row, dict):
                rows.append({k: (v or "") for k, v in row.items()})
        return (fieldnames, rows)


def _rewrite_export_csv_filtered(
    export_csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    export_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = export_csv_path.with_suffix(export_csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(export_csv_path)


def _export_has_url(export_rows: list[dict[str, str]], url: str) -> bool:
    u = (url or "").strip()
    if not u:
        return False
    for r in export_rows:
        if (r.get("webpage_url") or "").strip() == u:
            return True
    return False


def _meta_has_url(meta_index: dict[str, dict], url: str) -> str | None:
    u = (url or "").strip()
    if not u:
        return None
    for vid, obj in meta_index.items():
        if not isinstance(obj, dict):
            continue
        if (obj.get("webpage_url") or "").strip() == u:
            return vid
    return None


def _id_from_export_row(row: dict[str, str]) -> str | None:
    u = (row.get("webpage_url") or "").strip()
    if u:
        vid = _extract_id_from_url(u)
        if vid:
            return vid
    af = (row.get("audio_filename") or "").strip().replace("\\", "/")
    m = re.search(r"/audio/([^/]+)/", af)
    if m:
        return (m.group(1) or "").strip() or None
    return None


def write_ingest_summary(root: Path, audio_dir: Path) -> None:
    """本轮抓取结束后：统计含 timeline 的条目，并写出 urls 汇总（UTF-8）。"""
    try:
        from list_audio_without_timeline import has_timeline_rows
    except ImportError:

        def has_timeline_rows(p: Path) -> bool:  # type: ignore[misc,redef]
            if not p.is_file():
                return False
            try:
                with p.open("r", encoding="utf-8-sig", newline="") as f:
                    r = csv.DictReader(f)
                    if not r.fieldnames:
                        return False
                    for row in r:
                        if isinstance(row, dict) and any((v or "").strip() for v in row.values()):
                            return True
            except OSError:
                return False
            return False

    export_csv_path = root / "export.csv"
    fieldnames, rows = _load_export_csv_rows(export_csv_path)

    ids_on_disk: list[str] = []
    if audio_dir.is_dir():
        for d in sorted(audio_dir.iterdir(), key=lambda p: p.name.casefold()):
            if d.is_dir():
                ids_on_disk.append(d.name)

    ids_with_tl: list[str] = []
    for vid in ids_on_disk:
        if has_timeline_rows(audio_dir / vid / "timeline.csv"):
            ids_with_tl.append(vid)

    id_set_tl = frozenset(ids_with_tl)
    urls_tl: list[str] = []
    seen_tl_url = set()
    urls_all: list[str] = []
    seen_u = set()
    for row in rows:
        u = (row.get("webpage_url") or "").strip()
        if u and u not in seen_u:
            seen_u.add(u)
            urls_all.append(u)
        vid = _id_from_export_row(row)
        if vid and vid in id_set_tl and u and u not in seen_tl_url:
            seen_tl_url.add(u)
            urls_tl.append(u)

    stats_path = root / "stats_ingest.txt"
    urls_tl_path = root / "urls_with_timeline.txt"
    urls_all_path = root / "urls_from_export.txt"

    lines = [
        f"audio 子目录数: {len(ids_on_disk)}",
        f"含有效 timeline.csv 的 id 数: {len(ids_with_tl)}",
        f"export.csv 中不重复 URL 数: {len(urls_all)}",
        f"与「磁盘上已有 timeline」对应的 URL 数: {len(urls_tl)}",
        "",
        "含 timeline 的 id 列表:",
        *ids_with_tl,
    ]
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    urls_tl_path.write_text("\n".join(urls_tl) + ("\n" if urls_tl else ""), encoding="utf-8")
    urls_all_path.write_text("\n".join(urls_all) + ("\n" if urls_all else ""), encoding="utf-8")

    print(f"[STATS] 写入 {stats_path.name}、{urls_tl_path.name}、{urls_all_path.name}")


def _audio_folder_has_wav(per_id_dir: Path) -> bool:
    if not per_id_dir.exists() or not per_id_dir.is_dir():
        return False
    for p in per_id_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".wav":
            return True
    return False


def _purge_related_records(
    *,
    url: str,
    vid: str | None,
    export_csv_path: Path,
    meta_index_path: Path,
    audio_dir: Path,
) -> None:
    # 1) export.csv: remove rows matching webpage_url, or rows whose audio_filename points to audio/<vid>/
    fieldnames, rows = _load_export_csv_rows(export_csv_path)
    u = (url or "").strip()
    v = (vid or "").strip()

    def row_matches(r: dict[str, str]) -> bool:
        if u and (r.get("webpage_url") or "").strip() == u:
            return True
        if v:
            af = (r.get("audio_filename") or "").replace("\\", "/")
            if f"/audio/{v}/" in af or af.endswith(f"/{v}.wav") or af.endswith(f"/{v}.m4a") or af.endswith(f"/{v}.mp3"):
                return True
        return False

    kept = [r for r in rows if not row_matches(r)]
    if len(kept) != len(rows):
        _rewrite_export_csv_filtered(export_csv_path, fieldnames, kept)

    # 2) meta.json: remove by id, and also remove any entry whose webpage_url matches
    idx = _load_meta_index(meta_index_path)
    changed = False
    if v and v in idx:
        idx.pop(v, None)
        changed = True
    if u:
        to_del = [k for k, obj in idx.items() if isinstance(obj, dict) and (obj.get("webpage_url") or "").strip() == u]
        for k in to_del:
            idx.pop(k, None)
        changed = changed or bool(to_del)
    if changed:
        meta_index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) audio: delete per-id folder and stray files in audio_dir matching {vid}.*
    if v:
        per_id_dir = audio_dir / v
        if per_id_dir.exists() and per_id_dir.is_dir():
            shutil.rmtree(per_id_dir)
        for p in audio_dir.glob(f"{v}.*"):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="下载 YouTube 音频到本地，并导出标题/作者/上传日期与简介中的 timeline 时间戳"
    )
    ap.add_argument("--url", type=str, default=None, help="单个视频 URL")
    ap.add_argument("--url-file", type=Path, default=None, help="包含多个 URL 的文本文件（每行一个，# 开头视为注释）")
    ap.add_argument("--cookies", type=Path, default=None, help="Netscape cookies.txt（可选；需要登录态时填）")
    ap.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="直接从浏览器读取 cookies（可选）：例如 edge / chrome / firefox。与 --cookies 二选一",
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="元数据根目录（写 export.csv、meta.json、failed_urls.txt、stats_*.txt 等）",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（不填则 <root>\\audio）",
    )
    ap.add_argument(
        "--audio-format",
        type=str,
        default="wav",
        help="音频格式：wav/m4a/mp3/flac...（默认 wav）",
    )
    ap.add_argument(
        "--audio-quality",
        type=str,
        default="0",
        help="音频质量（yt-dlp 透传给 ffmpeg），默认 0（最佳）",
    )
    ap.add_argument(
        "--outtmpl",
        type=str,
        default="%(id)s.%(ext)s",
        help="输出文件名模板（相对 --audio-dir），默认 %%(id)s.%%(ext)s",
    )
    ap.add_argument(
        "--remote-ejs",
        type=str,
        default="github",
        help="EJS 组件来源（用于解决 YouTube n challenge）。默认 github；传空字符串可关闭",
    )
    ap.add_argument(
        "--js-runtimes",
        type=str,
        default=None,
        help="指定 JS 运行时（例如 deno 或 node）。不填则让 yt-dlp 自动探测",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="批量时遇到单条下载失败则跳过继续，并把失败 URL 追加写入 <root>\\failed_urls.txt",
    )
    ap.add_argument(
        "--require-timeline",
        action="store_true",
        help="若简介中解析不到任何时间轴则本视频不下载、不写 meta/export（仍会先按原逻辑做不完整清理）",
    )
    ap.add_argument(
        "--title-keywords",
        type=str,
        default=None,
        metavar="KW,KW2",
        help="英文逗号分隔多个关键词（每项内可含空格）；仅当标题子串匹配其一（忽略大小写）时才下载并写库；不传则不按标题过滤",
    )
    ap.add_argument(
        "--write-ingest-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="结束时写入数据清单统计；默认开启，可用 --no-write-ingest-summary 关闭",
    )

    args = ap.parse_args()
    if not args.url and not args.url_file:
        raise SystemExit("请至少提供 --url 或 --url-file")
    if args.cookies and args.cookies_from_browser:
        raise SystemExit("请在 --cookies 与 --cookies-from-browser 中二选一")

    urls = list(_iter_urls(args.url, args.url_file))
    if not urls:
        raise SystemExit("未找到任何 URL（检查 --url 或 --url-file）")

    title_keywords_lc = _parse_csv_keywords_arg(args.title_keywords)
    run_summary = RunSummary("collection.download_audio_and_timeline")

    cookies = args.cookies
    cookies_from_browser = (args.cookies_from_browser or "").strip() or None
    with tempfile.TemporaryDirectory(prefix="yt_cookies_") as td:
        if cookies is not None:
            cookies = sanitize_cookies_for_youtube(cookies, Path(td))

        audio_dir = args.audio_dir or (args.root / "audio")
        export_csv_path = args.root / "export.csv"
        meta_index_path = args.root / "meta.json"
        failed_urls_path = args.root / "failed_urls.txt"

        for u in urls:
            try:
                # ---- 新断点逻辑（按 url -> csv/meta -> id -> audio wav 完整性）----
                # 规则：先按 urls.txt 的 url 检查 export.csv 与 meta.json 是否都存在该 url；
                # 若都有，再用 id 检查 audio/<id>/ 里是否有 wav；三者都满足则直接跳过。
                fieldnames, export_rows = _load_export_csv_rows(export_csv_path)
                meta_idx = _load_meta_index(meta_index_path)
                export_has = _export_has_url(export_rows, u)
                meta_vid = _meta_has_url(meta_idx, u)
                expected_vid = meta_vid or _extract_id_from_url(u)

                if export_has and meta_vid and expected_vid:
                    per_id_dir = audio_dir / expected_vid
                    if _audio_folder_has_wav(per_id_dir):
                        print(f"[SKIP] {expected_vid}（csv/meta/url + audio wav 都存在）")
                        run_summary.add("skipped", item_id=expected_vid, reason="already_complete")
                        continue

                # 任一环节不满足：清理三处相关信息，然后重新下载记录
                if not (export_has and meta_vid and expected_vid and _audio_folder_has_wav(audio_dir / expected_vid)):
                    _purge_related_records(
                        url=u,
                        vid=expected_vid,
                        export_csv_path=export_csv_path,
                        meta_index_path=meta_index_path,
                        audio_dir=audio_dir,
                    )

                # 对 info 获取也加上 EJS 组件（通过环境/缓存通常即可；这里用一次 list-formats/下载时更关键）
                info = yt_dlp_info(u, cookies, cookies_from_browser)
                vid = _safe_video_id(info)
                meta_obj = _meta_object_from_info(info)
                description = info.get("description") or ""
                timeline = parse_timeline_from_description(description)
                title = (meta_obj.get("title") or info.get("title") or "").strip()

                if title_keywords_lc is not None and not _title_matches_any_keyword(title, title_keywords_lc):
                    print(f"[SKIP-FILTER] {vid} 标题不包含任一关键词: {args.title_keywords!r}")
                    run_summary.add("skipped", item_id=vid, reason="title_filter")
                    continue

                if args.require_timeline and not timeline:
                    print(f"[SKIP-FILTER] {vid} 简介中无可用时间轴（--require-timeline）")
                    run_summary.add("skipped", item_id=vid, reason="missing_timeline")
                    continue

                per_id_dir = audio_dir / vid
                per_id_dir.mkdir(parents=True, exist_ok=True)
                final_audio_path = per_id_dir / f"{vid}.{args.audio_format}"

                # 下载：如果目录里已有 wav 则不再下载（但此时仍会更新 meta/export，保证记录齐全）
                if not _audio_folder_has_wav(per_id_dir):
                    yt_dlp_download_audio(
                        u,
                        cookies,
                        cookies_from_browser,
                        audio_dir,
                        args.audio_format,
                        args.audio_quality,
                        args.outtmpl,
                        remote_ejs=(args.remote_ejs or None),
                        js_runtimes=args.js_runtimes,
                    )

                    downloaded = _find_downloaded_audio(audio_dir, vid, args.audio_format)
                    if downloaded is None:
                        raise RuntimeError(f"未找到下载后的音频文件：audio_dir={audio_dir} video_id={vid}")

                    moved_target = per_id_dir / f"{vid}.{downloaded.suffix.lstrip('.')}"
                    if downloaded.resolve() != moved_target.resolve():
                        try:
                            downloaded.replace(moved_target)
                        except OSError:
                            shutil.move(str(downloaded), str(moved_target))
                    final_audio_path = moved_target
                else:
                    # 目录里已有 wav：选一个用于写 export.csv 的 audio_filename
                    wavs = sorted([p for p in per_id_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
                    if wavs:
                        final_audio_path = wavs[0]

                if timeline:
                    timeline_csv_path = per_id_dir / "timeline.csv"
                    _write_timeline_csv(timeline_csv_path, vid, timeline)

                # 汇总文件（增量）
                _update_meta_index(meta_index_path, meta_obj)
                audio_filename_for_export = _relpath_or_name(final_audio_path, args.root)
                append_export_csv(
                    export_csv_path,
                    webpage_url=meta_obj.get("webpage_url"),
                    uploader=meta_obj.get("uploader"),
                    audio_filename=audio_filename_for_export,
                    title=meta_obj.get("title"),
                    timeline=timeline,
                )

                print(f"[OK] {vid}")
                print(f"  meta.json: {meta_index_path}")
                print(f"  export.csv: {export_csv_path}")
                print(f"  audio: {final_audio_path}")
                if timeline:
                    print(f"  timeline.csv: {per_id_dir / 'timeline.csv'}")
                run_summary.add("success", item_id=vid)
            except Exception as e:
                print(f"[ERR] {u}")
                print(f"  {e}")
                run_summary.add("failed", item_id=u, detail=e)
                if args.continue_on_error:
                    failed_urls_path.parent.mkdir(parents=True, exist_ok=True)
                    with failed_urls_path.open("a", encoding="utf-8") as f:
                        f.write(u + "\n")
                    continue
                run_summary.finish(args.root / "download_run_summary.json")
                raise

        if args.write_ingest_summary:
            write_ingest_summary(args.root, audio_dir)
        run_summary.finish(
            args.root / "download_run_summary.json",
            metadata={"requested_urls": len(urls), "audio_format": args.audio_format},
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
