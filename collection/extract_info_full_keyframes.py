"""
按 info_full.json 各条目的 wav 文件名（id_start_end）从 YouTube 截关键帧。

截帧时刻：原视频时间轴 [start, end] 内 start + offset_sec（默认 10s），过短则钳制在窗内。
输出：<figs-dir>/<wav stem>.png（与 info_full 中 wav 命名一致）。

本文件自包含 yt-dlp + ffmpeg 截帧逻辑，不依赖 trash 或其它 code 模块。
系统需已安装 yt-dlp、ffmpeg。

示例（批量推荐与 yt_frame_extract 一致：native-local + 按视频缓存）：
  python extract_info_full_keyframes.py \\
    --info-json ../info_full.json --figs-dir ../figs --meta ../meta.json \\
    --cookies-from-browser edge --js-runtimes deno --reuse-audio-download \\
    --audio-dir ../audio --skip-existing --continue-on-error
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from deepasmr_nspeech import RunSummary  # noqa: E402

# --- yt-dlp / ffmpeg 截帧（自包含）---

_DEFAULT_FFNET_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_FFMPEG_PROTO_WHITELIST = "file,crypto,data,http,https,tcp,tls,httpproxy"
_YT_COOKIE_DOMAIN_HINTS = ("youtube", "youtu.be", "googlevideo")
_YOUTUBE_COOKIE_DOMAIN_KEYWORDS = (
    "youtube.com",
    "google.com",
    "accounts.google.com",
    "googlevideo.com",
    "ytimg.com",
)

SnippetMode = Literal["ffmpeg-section", "native-local"]


def _run(cmd: list[str], *, subprocess_env: dict[str, str] | None = None) -> None:
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=subprocess_env,
    )
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"命令失败: {' '.join(cmd)}\n{msg}")


def _normalized_proxy_scheme(proxy: str) -> str:
    p = (proxy or "").strip()
    if not p:
        return p
    if not re.match(r"[\da-zA-Z\-]+://", p):
        return f"http://{p}"
    return p


def env_for_ytdlp_subprocess(proxy_cli: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if proxy_cli is not None:
        if str(proxy_cli).strip() == "":
            for k in _PROXY_ENV_KEYS:
                env.pop(k, None)
            return env
        p = _normalized_proxy_scheme(str(proxy_cli))
        if p:
            for k in _PROXY_ENV_KEYS:
                env[k] = p
        return env
    hp = env.get("HTTP_PROXY") or env.get("http_proxy")
    if hp:
        env.setdefault("HTTPS_PROXY", hp)
        env.setdefault("https_proxy", hp)
    return env


def cookie_header_from_netscape(cookie_path: Path, *, max_len: int = 3500) -> str | None:
    now_ts = time.time()
    pairs: list[str] = []
    try:
        text = cookie_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t") if "\t" in line else line.split()
        if len(cols) != 7:
            continue
        dom = cols[0].strip().lower().lstrip(".")
        if not any(h in dom for h in _YT_COOKIE_DOMAIN_HINTS):
            continue
        try:
            exp = int(float(cols[4]))
        except ValueError:
            continue
        if exp > 0 and float(exp) < now_ts:
            continue
        name, value = cols[5], cols[6]
        if name:
            pairs.append(f"{name}={value}")
        if len("; ".join(pairs)) > max_len:
            pairs.pop()
            break
    return "; ".join(pairs) if pairs else None


def _ffmpeg_section_downloader_args(cookie_file: Path | None, *, ua: str = _DEFAULT_FFNET_UA) -> list[str]:
    chunks: list[str] = ["-protocol_whitelist", _FFMPEG_PROTO_WHITELIST]
    ck = cookie_header_from_netscape(cookie_file) if cookie_file is not None else None
    if ck:
        block = "Cookie: " + ck + "\r\nUser-Agent: " + ua.strip() + "\r\n"
        chunks.extend(["-headers", block])
    return ["--downloader-args", f"ffmpeg_i1:{shlex.join(chunks)}"]


def sanitize_cookies_for_youtube(cookies_path: Path, tmp_dir: Path) -> Path:
    text = cookies_path.read_text(encoding="utf-8", errors="replace")
    out_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        cols = line.split("\t") if "\t" in line else line.split()
        if len(cols) != 7:
            continue
        domain = cols[0].strip()
        include_sub = cols[1].strip().upper()
        d = domain.lstrip(".").lower()
        if not any(k in d for k in _YOUTUBE_COOKIE_DOMAIN_KEYWORDS):
            continue
        if domain.startswith(".") and include_sub == "FALSE":
            cols[1] = "TRUE"
        out_lines.append("\t".join(cols))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "cookies_youtube_sanitized.txt"
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_path


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


def yt_dlp_download_video_snippet(
    url: str,
    cookies: Path | None,
    cookies_from_browser: str | None,
    output_template: str | Path,
    video_format: str,
    *,
    remote_ejs: str | None,
    js_runtimes: str | None,
    force_ipv4: bool,
    socket_timeout_sec: int,
    proxy: str | None,
    extractor_args: str | None,
) -> None:
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


def _trim_float(x: float) -> str:
    t = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return t if t else "0"


def _stamp_section_time(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s >= 3600:
        h, r = divmod(s, 3600.0)
        m, sec = divmod(r, 60.0)
        return f"{int(h)}:{int(m):02d}:{_trim_float(sec)}"
    if s >= 60:
        m, sec = divmod(s, 60.0)
        return f"{int(m)}:{_trim_float(sec)}"
    return _trim_float(s)


def secs_to_download_section(start_sec: float, end_sec: float) -> str:
    ss = float(start_sec)
    ee = float(end_sec)
    if ee <= ss:
        raise ValueError("end_sec 必须大于 start_sec")
    min_span = 0.05
    if ee - ss < min_span:
        ee = ss + min_span
    ls = _stamp_section_time(ss)
    le = _stamp_section_time(ee)
    if ls == le:
        ee = ss + max(min_span, 0.01)
        le = _stamp_section_time(ee)
    return f"*{ls}-{le}"


def capture_timestamp_in_window(cap_start: float, cap_end: float, seek_offset: float) -> float:
    lo = float(cap_start)
    hi = float(cap_end)
    if hi <= lo:
        hi = lo + 0.05
    span = hi - lo
    margin = max(1e-3, min(0.05, span * 0.05))
    t = lo + float(seek_offset)
    if t >= hi - margin:
        t = lo + max(margin, min(float(seek_offset), span * 0.5))
    if t >= hi - margin:
        t = lo + span * 0.5
    return min(max(t, lo + margin), hi - margin)


def run_ffmpeg(ff: list[str]) -> None:
    p = subprocess.run(ff, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 失败 ({p.returncode})\n{err}")


def _glob_media_file(parent: Path, pattern: str) -> Path | None:
    cands = sorted(
        (
            p
            for p in parent.glob(pattern)
            if p.is_file() and not p.name.endswith(".part") and p.stat().st_size > 0
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def find_media(tmp: Path, pattern: str) -> Path:
    p = _glob_media_file(tmp, pattern)
    if p is None:
        names = sorted(x.name for x in tmp.iterdir() if x.is_file())
        raise RuntimeError(f"未找到 {pattern}，目录内有: {names[:24]}")
    return p


def _ytdlp_transient(err_text: str) -> bool:
    t = err_text.lower()
    return any(
        x in t
        for x in (
            "ffmpeg exited",
            "-138",
            "-10054",
            "timed out",
            "timeout",
            "connection",
            "connection reset",
            "proxy",
            "remote end closed",
            "end of file",
            "ssl",
            "unexpected_eof",
            "tcp://",
        )
    )


def frame_cache_dir_for_video(
    figs_dir: Path,
    video_id: str,
    snippet_mode: SnippetMode,
    *,
    audio_dir: Path | None,
    cache_parent: Path | None,
) -> Path | None:
    """与 yt_frame_extract 批量一致：native-local 时每个视频 id 一个 .ytdlp_frame_cache。"""
    if snippet_mode != "native-local":
        return None
    if cache_parent is not None:
        return (cache_parent / video_id).resolve()
    if audio_dir is not None:
        return (audio_dir / video_id / ".ytdlp_frame_cache").resolve()
    return (figs_dir / ".ytdlp_frame_cache" / video_id).resolve()


def extract_one_frame(
    watch_url: str,
    *,
    at_sec: float,
    output_png: Path,
    cookies: Path | None,
    cookies_browser: str | None,
    margin_before: float,
    margin_after: float,
    remote_ejs: str | None,
    js_runtimes: str | None,
    video_format: str,
    local_video_format: str,
    snippet_mode: SnippetMode,
    cache_dir: Path | None,
    force_kf: bool,
    ffmpeg_accurate_seek: bool,
    force_ipv4: bool,
    socket_timeout_sec: int,
    proxy: str | None,
    proxy_env_override: str | None,
    section_extra_ffmpeg_i1: bool,
    yt_retries: int,
) -> None:
    t0 = max(0.0, float(at_sec) - float(margin_before))
    t1 = float(at_sec) + float(margin_after)
    inside_clip = float(at_sec) - t0
    extractor_fallbacks: list[str | None] = [None, "youtube:player_client=android", "youtube:player_client=web"]

    with tempfile.TemporaryDirectory(prefix="yt_clip_") as tdir:
        td = Path(tdir)
        cookie_file = None
        if cookies is not None and not cookies_browser:
            cookie_file = sanitize_cookies_for_youtube(cookies, td)
        video_disk_dir = cache_dir if (snippet_mode == "native-local" and cache_dir is not None) else td

        if snippet_mode == "native-local":
            video_disk_dir.mkdir(parents=True, exist_ok=True)
            seek_at = float(at_sec)
            vid_ready = _glob_media_file(video_disk_dir, "snippet_src.*")
            if vid_ready is None:
                last_exc_native: Exception | None = None
                for attempt in range(max(1, int(yt_retries))):
                    ea = extractor_fallbacks[min(attempt, len(extractor_fallbacks) - 1)]
                    try:
                        yt_dlp_download_video_snippet(
                            watch_url,
                            cookie_file,
                            cookies_browser,
                            video_disk_dir / "snippet_src.%(ext)s",
                            local_video_format,
                            remote_ejs=remote_ejs,
                            js_runtimes=js_runtimes,
                            force_ipv4=force_ipv4,
                            socket_timeout_sec=socket_timeout_sec,
                            proxy=proxy,
                            extractor_args=ea,
                        )
                        last_exc_native = None
                        break
                    except RuntimeError as e:
                        last_exc_native = e
                        if not _ytdlp_transient(str(e)) or attempt >= yt_retries - 1:
                            raise
                        time.sleep(2.0 * float(attempt + 1))
                vid_ready = find_media(video_disk_dir, "snippet_src.*")
                if last_exc_native is not None and vid_ready is None:
                    raise RuntimeError(
                        "native-local 未生成 snippet_src.*（可能被 --no-overwrites 跳过与已有半成品冲突；"
                        "可删缓存目录后重试）\n"
                        f"{last_exc_native}"
                    ) from last_exc_native
            output_png.parent.mkdir(parents=True, exist_ok=True)
            ff = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
            seek_s = str(seek_at)
            if ffmpeg_accurate_seek:
                ff += ["-i", str(vid_ready), "-ss", seek_s, "-frames:v", "1", "-q:v", "2"]
            else:
                ff += ["-ss", seek_s, "-i", str(vid_ready), "-frames:v", "1", "-q:v", "2"]
            ff.append(str(output_png))
            run_ffmpeg(ff)
            return

        outtmpl = td / "clip.%(ext)s"
        subproc_env = env_for_ytdlp_subprocess(proxy_env_override)
        last_exc: Exception | None = None
        for attempt in range(max(1, int(yt_retries))):
            ea = extractor_fallbacks[min(attempt, len(extractor_fallbacks) - 1)]
            cmd: list[str] = [
                "yt-dlp",
                "--no-playlist",
                "-f",
                video_format,
                "--download-sections",
                secs_to_download_section(t0, t1),
                "-o",
                str(outtmpl),
            ]
            if force_kf:
                cmd.append("--force-keyframes-at-cuts")
            ytdlp_prepend_standard_options(
                cmd,
                cookies=cookie_file,
                cookies_from_browser=cookies_browser,
                remote_ejs=remote_ejs,
                js_runtimes=js_runtimes,
                force_ipv4=force_ipv4,
                socket_timeout_sec=socket_timeout_sec,
                proxy=proxy,
            )
            if section_extra_ffmpeg_i1:
                cf = cookie_file if cookies_browser is None else None
                cmd.extend(_ffmpeg_section_downloader_args(cf))
            if ea:
                cmd.extend(["--extractor-args", ea])
            cmd.append(watch_url)
            try:
                _run(cmd, subprocess_env=subproc_env)
                last_exc = None
                break
            except RuntimeError as e:
                last_exc = e
                if not _ytdlp_transient(str(e)) or attempt >= yt_retries - 1:
                    raise
                time.sleep(2.0 * float(attempt + 1))
        if last_exc is not None:
            raise last_exc

        vidfile = find_media(td, "clip.*")
        output_png.parent.mkdir(parents=True, exist_ok=True)
        ff = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        if ffmpeg_accurate_seek:
            ff += ["-i", str(vidfile), "-ss", str(inside_clip), "-frames:v", "1", "-q:v", "2"]
        else:
            ff += ["-ss", str(inside_clip), "-i", str(vidfile), "-frames:v", "1", "-q:v", "2"]
        ff.append(str(output_png))
        run_ffmpeg(ff)


# --- info_full 任务 ---


def parse_fname_num(tok: str) -> float:
    t = (tok or "").strip()
    if not t:
        raise ValueError("empty time token")
    return float(t.replace("p", "."))


def parse_wav_stem(stem: str) -> Tuple[str, float, float]:
    s = (stem or "").strip()
    if not s:
        raise ValueError("empty stem")
    parts = s.split("_")
    if len(parts) < 3:
        raise ValueError(f"stem 至少需 id_start_end 三段: {stem!r}")
    video_id = "_".join(parts[:-2])
    if not video_id:
        raise ValueError(f"无法解析 video_id: {stem!r}")
    return video_id, parse_fname_num(parts[-2]), parse_fname_num(parts[-1])


def load_info_full_entries(info_path: Path) -> List[Dict[str, Any]]:
    with info_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        rows = raw["data"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError("info_full.json 需为 {data: [...]} 或顶层数组")
    return [row for row in rows if isinstance(row, dict)]


def load_meta_watch_urls(meta_path: Path) -> Dict[str, str]:
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    urls: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return urls
    for k, v in raw.items():
        if isinstance(v, dict):
            vid = (v.get("id") or k or "").strip()
            u = (v.get("webpage_url") or "").strip()
            if vid and u.startswith("http"):
                urls[vid] = u
    return urls


def watch_url_for_id(video_id: str, meta_urls: Dict[str, str]) -> str:
    return meta_urls.get(video_id) or f"https://www.youtube.com/watch?v={video_id}"


def iter_tasks(
    entries: List[Dict[str, Any]],
    *,
    figs_dir: Path,
    meta_urls: Dict[str, str],
    offset_sec: float,
    stem_filter: Optional[frozenset[str]] = None,
) -> List[Tuple[Path, str, str, str, float, float, float]]:
    tasks: List[Tuple[Path, str, str, str, float, float, float]] = []
    seen_stems: set[str] = set()
    for row in entries:
        wav = (row.get("wav") or "").strip()
        if not wav:
            continue
        stem = Path(wav).stem
        if stem_filter is not None and stem not in stem_filter:
            continue
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        try:
            vid, start, end = parse_wav_stem(stem)
        except ValueError as e:
            print(f"[SKIP-PARSE] {stem}: {e}")
            continue
        cap_at = capture_timestamp_in_window(start, end, offset_sec)
        tasks.append((figs_dir / f"{stem}.png", watch_url_for_id(vid, meta_urls), stem, vid, start, end, cap_at))
    tasks.sort(key=lambda t: (t[3], t[2]))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(
        description="按 info_full.json 为各 wav 段截 YouTube 关键帧到 figs/（截帧逻辑对齐 yt_frame_extract.py）"
    )
    ap.add_argument("--info-json", type=Path, default=_repo_root / "info_full.json")
    ap.add_argument("--figs-dir", type=Path, default=_repo_root / "figs")
    ap.add_argument("--meta", type=Path, default=_repo_root / "meta.json")
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="与 yt_frame_extract 一致：native-local 时优先用 <audio-dir>/<id>/.ytdlp_frame_cache 复用已下载视频",
    )
    ap.add_argument("--offset-sec", type=float, default=10.0, help="相对片段 start 的截帧秒数，默认 10")
    ap.add_argument("--cookies", type=Path, default=None)
    ap.add_argument("--cookies-from-browser", type=str, default=None)
    ap.add_argument("--remote-ejs", type=str, default="github")
    ap.add_argument("--js-runtimes", type=str, default=None)
    ap.add_argument(
        "--snippet-mode",
        choices=("ffmpeg-section", "native-local"),
        default="ffmpeg-section",
        help=(
            "ffmpeg-section：--download-sections（与 yt_frame_extract 默认相同）。"
            "native-local：yt-dlp 整段低清下载后本地 seek（与 --reuse-audio-download 相同）。"
        ),
    )
    ap.add_argument(
        "--reuse-audio-download",
        action="store_true",
        help="等价于 --snippet-mode native-local（与下载音频 / yt_frame_extract 拉流路径一致）",
    )
    ap.add_argument(
        "--video-format",
        type=str,
        default="bv*[height<=720]/bv*[height<=1080]/bv*/bestvideo/best",
        help="snippet-mode=ffmpeg-section 时的 yt-dlp -f",
    )
    ap.add_argument(
        "--local-video-format",
        type=str,
        default="bv*[height<=480]/wv*[height<=480]/wv*/worstvideo/worst",
        help="snippet-mode=native-local 时的 yt-dlp -f",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="native-local 时缓存根目录，实际为 <cache-dir>/<video_id>/snippet_src.*；未设则用 figs/.ytdlp_frame_cache 或 audio-dir",
    )
    ap.add_argument("--margin-before", type=float, default=2.5)
    ap.add_argument("--margin-after", type=float, default=6.0)
    ap.add_argument(
        "--no-section-ffmpeg-i1-headers",
        action="store_true",
        help="关闭 ffmpeg-section 的 downloader-args ffmpeg_i1（protocol_whitelist + Cookie 头）",
    )
    ap.add_argument("--force-keyframes-at-cuts", action="store_true")
    ap.add_argument("--accurate-seek-ffmpeg", action="store_true")
    ap.add_argument(
        "--force-ipv4",
        action="store_true",
        default=sys.platform.startswith("win"),
        help="yt-dlp/下游强制 IPv4（Windows 默认开启，与 yt_frame_extract 一致）",
    )
    ap.add_argument("--no-force-ipv4", dest="force_ipv4", action="store_false")
    ap.add_argument("--socket-timeout", type=int, default=120, metavar="SEC", dest="socket_timeout_sec")
    ap.add_argument(
        "--proxy",
        type=str,
        default=None,
        help='传给 yt-dlp；可用 --proxy "" 清空环境代理（与 yt_frame_extract 一致）',
    )
    ap.add_argument("--yt-retries", type=int, default=3)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stems", type=str, default=None, help="逗号分隔 wav stem")
    args = ap.parse_args()

    if args.reuse_audio_download:
        args.snippet_mode = "native-local"

    if args.cookies and args.cookies_from_browser:
        print("[ERR] --cookies 与 --cookies-from-browser 二选一", file=sys.stderr)
        return 2

    info_path = args.info_json.resolve()
    figs_dir = args.figs_dir.resolve()
    if not info_path.is_file():
        print(f"[ERR] 不存在: {info_path}", file=sys.stderr)
        return 2

    entries = load_info_full_entries(info_path)
    meta_urls = load_meta_watch_urls(args.meta.resolve())
    stem_filter = (
        frozenset(x.strip() for x in args.stems.split(",") if x.strip()) if args.stems else None
    )
    tasks = iter_tasks(
        entries,
        figs_dir=figs_dir,
        meta_urls=meta_urls,
        offset_sec=float(args.offset_sec),
        stem_filter=stem_filter,
    )
    if args.limit > 0:
        tasks = tasks[: int(args.limit)]

    figs_dir.mkdir(parents=True, exist_ok=True)
    ok, skip, err = 0, 0, 0
    run_summary = RunSummary("collection.extract_keyframes")
    audio_dir = args.audio_dir.resolve() if args.audio_dir is not None else None
    cache_parent = args.cache_dir.resolve() if args.cache_dir is not None else None
    proxy_kw = args.proxy.strip() if args.proxy and args.proxy.strip() else args.proxy
    cookie_p = args.cookies.resolve() if args.cookies else None
    cfb = (args.cookies_from_browser or "").strip() or None
    print(
        f"[INFO] entries={len(entries)} tasks={len(tasks)} figs_dir={figs_dir} "
        f"offset={args.offset_sec}s snippet_mode={args.snippet_mode} force_ipv4={args.force_ipv4}"
    )
    if args.snippet_mode == "native-local":
        cache_hint = (
            str(audio_dir / "<id>/.ytdlp_frame_cache")
            if audio_dir
            else str(figs_dir / ".ytdlp_frame_cache/<id>")
        )
        print(f"[INFO] native-local 按视频缓存: {cache_hint}")

    extract_kw_base = dict(
        cookies=cookie_p,
        cookies_browser=cfb,
        margin_before=args.margin_before,
        margin_after=args.margin_after,
        remote_ejs=(args.remote_ejs or "").strip() or None,
        js_runtimes=args.js_runtimes,
        video_format=args.video_format,
        local_video_format=args.local_video_format,
        snippet_mode=args.snippet_mode,
        force_kf=args.force_keyframes_at_cuts,
        ffmpeg_accurate_seek=args.accurate_seek_ffmpeg,
        force_ipv4=bool(args.force_ipv4),
        socket_timeout_sec=int(args.socket_timeout_sec),
        proxy=proxy_kw if proxy_kw != "" else "",
        proxy_env_override=args.proxy,
        section_extra_ffmpeg_i1=not args.no_section_ffmpeg_i1_headers,
        yt_retries=max(1, int(args.yt_retries)),
    )

    for out_png, url, stem, vid, start, end, cap_at in tasks:
        if out_png.is_file() and out_png.stat().st_size > 0 and args.skip_existing and not args.force:
            skip += 1
            run_summary.add("skipped", item_id=stem, reason="already_exists")
            print(f"[SKIP] {out_png.name}")
            continue
        if args.dry_run:
            print(f"[DRY] {out_png.name}  {vid} [{start:.3f},{end:.3f}] cap@{cap_at:.3f}s  {url}")
            ok += 1
            run_summary.add("success", item_id=stem, reason="dry_run")
            continue
        try:
            cache_dir = frame_cache_dir_for_video(
                figs_dir,
                vid,
                args.snippet_mode,
                audio_dir=audio_dir,
                cache_parent=cache_parent,
            )
            extract_one_frame(
                url,
                at_sec=cap_at,
                output_png=out_png,
                cache_dir=cache_dir,
                **extract_kw_base,
            )
            ok += 1
            run_summary.add("success", item_id=stem)
            print(f"[OK] {out_png.name}  cap={cap_at:.3f}s")
        except Exception as e:
            err += 1
            run_summary.add("failed", item_id=stem, detail=e)
            print(f"[ERR] {stem}: {e}")
            if not args.continue_on_error:
                run_summary.finish(figs_dir / "keyframe_run_summary.json")
                return 1

    run_summary.finish(
        figs_dir / "keyframe_run_summary.json",
        metadata={"input_entries": len(entries), "scheduled_tasks": len(tasks)},
    )
    print(f"[SUMMARY] ok/dry={ok} skip={skip} err={err} total={len(tasks)}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
