"""
用 yt-dlp + ffmpeg 对 YouTube 截单帧 PNG。两种截取策略：

1) ffmpeg-section（默认）：yt-dlp --download-sections，由 ffmpeg **直连 CDN URL** 切一小段后再本地截帧。
   yt-dlp 内置 FFmpegFD 已对输入 URL 写入 -cookies/-headers，但若仍报 -138：
   • 本子进程会为 yt-dlp 同步设置 HTTPS_PROXY=http(s)…（对齐官方仅写 HTTP_PROXY 的行为）；
   • 在用 --cookies 文件时可选再注入 --downloader-args ffmpeg_i1:...（protocol_whitelist + Cookie/User-Agent）。
2) native-local：整段低清 yt-dlp 下载后本地 seek 截帧（见前文）。

依赖：PATH 上有 yt-dlp、ffmpeg。
也可 --auto-audio 扫描 <audio-dir>/*/timeline.csv 批量截图（默认 audio-dir = <root>/audio）；frames 已满则跳过，不全则补。
若存在 <id>/segment_label_parse.json 且段内有 clean_start/clean_end，则按该窗截帧与命名（否则仍用 timeline 的 start/end）。
另可 --auto-segments：扫描 <segments-dir>/*/info.json，按 clean 窗截帧，PNG 写入该子目录 keyframe.png（在 export_clean_audio_segments 之后使用）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Sequence

from segment_layout import INFO_JSON, KEYFRAME_PNG, parse_clean_start_clean_end_field


# 与 yt-dlp 拉到 info 里的 http_headers User-Agent 同档即可，供 ffmpeg -headers 与站点行为对齐。
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

# ffmpeg 经 HTTP_PROXY/HTTPS_PROXY 出站时会启用 httpproxy 协议；白名单若不包含会直接报错。
_FFMPEG_PROTO_WHITELIST = "file,crypto,data,http,https,tcp,tls,httpproxy"


def _normalized_proxy_scheme(proxy: str) -> str:
    p = (proxy or "").strip()
    if not p:
        return p
    if not re.match(r"[\da-zA-Z\-]+://", p):
        return f"http://{p}"
    return p


def env_for_ytdlp_subprocess(proxy_cli: str | None) -> dict[str, str]:
    """让 yt-dlp 及其 fork 出的 ffmpeg 进程继承更合理的代理变量（尤其对 HTTPS CDN）。"""
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


_YT_COOKIE_DOMAIN_HINTS = ("youtube", "youtu.be", "googlevideo")


def cookie_header_from_netscape(cookie_path: Path, *, max_len: int = 3500) -> str | None:
    """从 Netscape 拼接 Cookie 头；仅含与播放相关的域，减小长度、避免超长命令行。"""
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
        if not name:
            continue
        pairs.append(f"{name}={value}")
        joined = "; ".join(pairs)
        if len(joined) > max_len:
            pairs.pop()
            break
    if not pairs:
        return None
    return "; ".join(pairs)


def _ffmpeg_section_downloader_args(cookie_file: Path | None, *, ua: str = _DEFAULT_FFNET_UA) -> list[str]:
    chunks: list[str] = ["-protocol_whitelist", _FFMPEG_PROTO_WHITELIST]
    ck = cookie_header_from_netscape(cookie_file) if cookie_file is not None else None
    if ck:
        block = "Cookie: " + ck + "\r\nUser-Agent: " + ua.strip() + "\r\n"
        chunks.extend(["-headers", block])
    joined = shlex.join(chunks)
    return ["--downloader-args", f"ffmpeg_i1:{joined}"]


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


def _stamp_section_time(seconds: float) -> str:
    """与 yt-dlp parse_duration 兼容：可分秒小数，避免整数秒碰撞。"""
    s = max(0.0, float(seconds))
    if s >= 3600:
        h, r = divmod(s, 3600.0)
        m, sec = divmod(r, 60.0)
        return f"{int(h)}:{int(m):02d}:{_trim_float(sec)}"
    if s >= 60:
        m, sec = divmod(s, 60.0)
        return f"{int(m)}:{_trim_float(sec)}"
    return _trim_float(s)


def _trim_float(x: float) -> str:
    t = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return t if t else "0"


def secs_to_download_section(start_sec: float, end_sec: float) -> str:
    """yt-dlp --download-sections 时间范围字符串，前缀 * 表示按时间而非章节名。"""
    ss = float(start_sec)
    ee = float(end_sec)
    if ee <= ss:
        raise ValueError("end_sec 必须大于 start_sec")

    min_span = 0.05  # 50ms：极短分段也保证 start<end（parse_duration 可读）
    if ee - ss < min_span:
        ee = ss + min_span

    ls = _stamp_section_time(ss)
    le = _stamp_section_time(ee)
    if ls == le:
        ee = ss + max(min_span, 0.01)
        le = _stamp_section_time(ee)

    return f"*{ls}-{le}"


def safe_fname_num(x: float) -> str:
    xf = float(x)
    iv = int(xf)
    return str(iv) if abs(xf - iv) < 1e-6 else str(xf).replace(".", "p")


def bad_filename(s: str) -> bool:
    return bool(re.search(r'[<>:"/\\|?*]', s))


def segment_output_png_name(vid: str, start_s: float, end_s: float) -> str | None:
    png_name = f"{vid}_{safe_fname_num(start_s)}_{safe_fname_num(end_s)}.png"
    if bad_filename(png_name):
        return None
    return png_name


DEFAULT_SEGMENT_JSON_NAME = "segment_label_parse.json"


def _json_segments_for_timeline_bounds(
    raw_list: list[Any],
    st: float,
    en: float,
    *,
    match_eps_sec: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for js in raw_list:
        if not isinstance(js, dict):
            continue
        jst = js.get("start_sec")
        jen = js.get("end_sec")
        if jst is None or jen is None:
            continue
        try:
            if abs(float(jst) - st) <= match_eps_sec and abs(float(jen) - en) <= match_eps_sec:
                out.append(js)
        except (TypeError, ValueError):
            continue

    def _sk(d: dict[str, Any]) -> tuple[float, float]:
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

    out.sort(key=_sk)
    return out


def resolve_capture_windows_from_parse_json(
    id_dir: Path,
    timeline_segs: Sequence[tuple[str, float, float]],
    *,
    segment_json_name: str = DEFAULT_SEGMENT_JSON_NAME,
    match_eps_sec: float = 0.51,
) -> list[tuple[str, float, float]]:
    """
    对每个 timeline 段，若在 segment_label_parse.json 中找到同 start_sec/end_sec 的条目且含有效 clean_*，
    则返回若干 (vid, clean_start, clean_end)（同一 timeline 行对应多段 ruptures 时展开多行）；
    否则单行 (vid, timeline_start, timeline_end)。
    """
    json_path = id_dir / segment_json_name
    if not json_path.is_file():
        return [(v, float(s), float(e)) for v, s, e in timeline_segs]

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [(v, float(s), float(e)) for v, s, e in timeline_segs]

    raw_list = payload.get("segments")
    if not isinstance(raw_list, list):
        return [(v, float(s), float(e)) for v, s, e in timeline_segs]

    out: list[tuple[str, float, float]] = []
    for vid, st, en in timeline_segs:
        st_f, en_f = float(st), float(en)
        matches = _json_segments_for_timeline_bounds(
            raw_list, st_f, en_f, match_eps_sec=match_eps_sec
        )
        cleaned: list[tuple[float, float]] = []
        for js in matches:
            c0, c1 = js.get("clean_start"), js.get("clean_end")
            if c0 is None or c1 is None:
                continue
            try:
                cs, ce = float(c0), float(c1)
            except (TypeError, ValueError):
                continue
            if ce > cs:
                cleaned.append((cs, ce))
        if not cleaned:
            out.append((vid, st_f, en_f))
        else:
            for cs, ce in cleaned:
                out.append((vid, cs, ce))
    return out


def load_segment_folder_job(seg_dir: Path) -> tuple[str, float, float] | None:
    """读取 <segments>/<id>_…/info.json，返回 (video_id, clean_start, clean_end)。"""
    info_path = seg_dir / INFO_JSON
    if not info_path.is_file():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(info, dict):
        return None
    vid = (info.get("id") or "").strip()
    key = info.get("clean_start_clean_end")
    if not vid or not isinstance(key, str):
        return None
    try:
        cs, ce = parse_clean_start_clean_end_field(key)
    except ValueError:
        return None
    if ce <= cs:
        return None
    return vid, cs, ce


def count_segment_keyframes_existing(segments_root: Path) -> tuple[int, int]:
    """统计 segments 下应有 keyframe.png 数及已存在非空文件数。"""
    expected = done = 0
    if not segments_root.is_dir():
        return 0, 0
    for seg_dir in sorted(segments_root.iterdir(), key=lambda p: p.name.casefold()):
        if not seg_dir.is_dir():
            continue
        if load_segment_folder_job(seg_dir) is None:
            continue
        expected += 1
        outp = seg_dir / KEYFRAME_PNG
        if outp.is_file() and outp.stat().st_size > 0:
            done += 1
    return done, expected


def batch_from_segments_dir(
    segments_root: Path,
    audio_parent_for_cache: Path,
    *,
    cookies: Path | None,
    cookies_browser: str | None,
    seek_offset: float,
    margin_before: float,
    margin_after: float,
    remote_ejs: str | None,
    js_runtimes: str | None,
    snippet_mode: SnippetMode,
    video_format: str,
    local_video_format: str,
    force_kf: bool,
    ffmpeg_accurate_seek: bool,
    force_png: bool,
    force_ipv4: bool,
    socket_timeout_sec: int,
    proxy: str | None,
    proxy_env_override: str | None,
    section_extra_ffmpeg_i1: bool,
    yt_retries: int,
) -> tuple[int, int]:
    """按 segments/*/info.json 的 clean 窗截帧，PNG 写入该子目录下的 keyframe.png。"""
    ok = err = 0
    watch = "https://www.youtube.com/watch?v={}"
    if not segments_root.is_dir():
        return 0, 0

    for seg_dir in sorted(segments_root.iterdir(), key=lambda p: p.name.casefold()):
        if not seg_dir.is_dir():
            continue
        job = load_segment_folder_job(seg_dir)
        if job is None:
            continue
        vid, cap_s, cap_e = job
        outp = seg_dir / KEYFRAME_PNG
        if outp.exists() and outp.stat().st_size > 0 and not force_png:
            continue
        t_cap = capture_timestamp_in_window(cap_s, cap_e, seek_offset)
        try:
            frame_cache = (
                None
                if snippet_mode != "native-local"
                else (audio_parent_for_cache / vid / ".ytdlp_frame_cache").resolve()
            )
            extract_one_frame(
                watch.format(vid),
                at_sec=t_cap,
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
                force_kf=force_kf,
                ffmpeg_accurate_seek=ffmpeg_accurate_seek,
                force_ipv4=force_ipv4,
                socket_timeout_sec=socket_timeout_sec,
                proxy=proxy,
                proxy_env_override=proxy_env_override,
                section_extra_ffmpeg_i1=section_extra_ffmpeg_i1,
                yt_retries=yt_retries,
            )
            ok += 1
            print(f"[OK] {outp}")
        except Exception as e:
            err += 1
            print(f"[ERR] {seg_dir.name} @ {t_cap}s: {e}")
    return ok, err


def capture_timestamp_in_window(cap_start: float, cap_end: float, seek_offset: float) -> float:
    """在 [cap_start, cap_end] 内取 start+seek_offset，越界则压缩到中点附近。"""
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


def read_timeline_segments(timeline_csv: Path, folder_video_id: str) -> list[tuple[str, float, float]]:
    """与 batch_from_timeline 相同规则解析 timeline.csv。"""
    if not timeline_csv.is_file():
        raise FileNotFoundError(str(timeline_csv))
    need_cols = frozenset({"video_id", "start", "end"})
    with timeline_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("timeline 无表头")
        hset = {(h or "").strip() for h in r.fieldnames if h}
        if not need_cols.issubset(hset):
            raise ValueError("timeline.csv 需含 video_id、start、end（请先 normalize）")
        fn = {(h or "").strip(): h for h in r.fieldnames if h}
        segs: list[tuple[str, float, float]] = []
        for row in r:
            if not isinstance(row, dict):
                continue
            vid = (row.get(fn["video_id"], "") or "").strip() or folder_video_id
            st = (row.get(fn["start"], "") or "").strip()
            en = (row.get(fn["end"], "") or "").strip()
            if not st or not en:
                continue
            segs.append((vid, float(st), float(en)))
    return segs


def discover_audio_ids_with_timeline(audio_dir: Path) -> list[str]:
    """audio 下二级目录中存在有效 timeline.csv 的文件夹名（即 batch-id）。"""
    try:
        from list_audio_without_timeline import has_timeline_rows  # noqa: WPS433
    except ImportError:

        def has_timeline_rows(csv_path: Path) -> bool:  # type: ignore[misc,redef]
            if not csv_path.is_file():
                return False
            try:
                return len(read_timeline_segments(csv_path, csv_path.parent.name)) > 0
            except Exception:
                return False

    ids: list[str] = []
    for entry in sorted(audio_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not entry.is_dir():
            continue
        if has_timeline_rows(entry / "timeline.csv"):
            ids.append(entry.name)
    return ids


def count_expected_frames_existing(
    audio_parent: Path,
    batch_id: str,
    *,
    frames_subdir: str,
    segs_capture: list[tuple[str, float, float]],
) -> tuple[int, int]:
    """
    按与 batch 一致的命名规则，统计应有的 PNG 数及已存在且非空的数量。
    segs_capture：每项为 (vid, cap_start, cap_end)，与截帧使用的窗口一致（可为 clean 窗）。
    audio_parent：各视频 id 子目录的父目录（与全脚本统一的 --audio-dir 一致）。
    非法文件名的段落不计入 expected（与批量循环中 err+=1 且不生成文件一致）。
    """
    out_root = audio_parent / batch_id / frames_subdir
    expected = 0
    done = 0
    for vid, start_s, end_s in segs_capture:
        name = segment_output_png_name(vid, start_s, end_s)
        if name is None:
            continue
        expected += 1
        p = out_root / name
        if p.is_file() and p.stat().st_size > 0:
            done += 1
    return done, expected


def run_ffmpeg(ff: list[str]) -> None:
    p = subprocess.run(ff, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 失败 ({p.returncode})\n{err}")


def find_snippet_src_media(video_dir: Path, prefix: str = "snippet_src") -> Path | None:
    cands = [
        p
        for p in video_dir.glob(f"{prefix}.*")
        if p.is_file()
        and not p.name.endswith(".part")
        and not p.name.endswith(".temp")
        and p.stat().st_size > 0
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_clip_media(tmp: Path) -> Path:
    files = sorted(
        (p for p in tmp.glob("clip.*") if p.is_file() and not p.name.endswith(".part") and p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        names = sorted(p.name for p in tmp.iterdir() if p.is_file())
        raise RuntimeError(f"未生成片段文件 clip.*，其中有: {names[:24]}")
    return files[0]


def _ytdlp_transient(err_text: str) -> bool:
    t = err_text.lower()
    return any(
        x in t
        for x in (
            "ffmpeg exited",
            "-138",
            "timed out",
            "timeout",
            "connection",
            "proxy",
            "remote end closed",
            "tcp://",
        )
    )


SnippetMode = Literal["ffmpeg-section", "native-local"]


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
    from download_yt_audio_and_timeline import (
        sanitize_cookies_for_youtube,
        yt_dlp_download_video_for_local_snippet,
        ytdlp_prepend_standard_options,
    )

    t0 = max(0.0, float(at_sec) - float(margin_before))
    t1 = float(at_sec) + float(margin_after)
    inside_clip = float(at_sec) - t0
    last_exc: Exception | None = None

    with tempfile.TemporaryDirectory(prefix="yt_clip_") as tdir:
        td = Path(tdir)

        cookie_file = None
        if cookies is not None and not cookies_browser:
            cookie_file = sanitize_cookies_for_youtube(cookies, td)

        video_disk_dir = cache_dir if (snippet_mode == "native-local" and cache_dir is not None) else td

        # mweb 常需 PO Token；失败时先试 android/web，减少 403/异常流形态
        extractor_fallbacks: list[str | None] = [
            None,
            "youtube:player_client=android",
            "youtube:player_client=web",
        ]

        if snippet_mode == "native-local":
            video_disk_dir.mkdir(parents=True, exist_ok=True)
            seek_at = float(at_sec)
            vid_ready = find_snippet_src_media(video_disk_dir)

            if vid_ready is None:
                last_exc_native: Exception | None = None
                for attempt in range(max(1, int(yt_retries))):
                    ea = extractor_fallbacks[min(attempt, len(extractor_fallbacks) - 1)]
                    try:
                        yt_dlp_download_video_for_local_snippet(
                            watch_url,
                            cookie_file,
                            cookies_browser,
                            str(video_disk_dir / "snippet_src.%(ext)s"),
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

                vid_ready = find_snippet_src_media(video_disk_dir)
                last_exc_msg = ""
                if last_exc_native is not None:
                    last_exc_msg = f"\n末次 yt-dlp: {last_exc_native}"
                if vid_ready is None:
                    raise RuntimeError(
                        "native-local：未生成 snippet_src.*（可能被 --no-overwrites 跳过与已有半成品冲突；"
                        "删除该目录下缓存后重试，或检查磁盘与格式 --local-video-format）"
                        + last_exc_msg
                    )

            vidfile = vid_ready

            output_png.parent.mkdir(parents=True, exist_ok=True)
            ff = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
            seek_s = str(seek_at)
            if ffmpeg_accurate_seek:
                ff += ["-i", str(vidfile), "-ss", seek_s, "-frames:v", "1", "-q:v", "2"]
            else:
                ff += ["-ss", seek_s, "-i", str(vidfile), "-frames:v", "1", "-q:v", "2"]
            ff.append(str(output_png))
            run_ffmpeg(ff)
            return

        outtmpl = td / "clip.%(ext)s"
        subproc_env = env_for_ytdlp_subprocess(proxy_env_override)
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

        vidfile = find_clip_media(td)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        ff = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
        if ffmpeg_accurate_seek:
            ff += ["-i", str(vidfile), "-ss", str(inside_clip), "-frames:v", "1", "-q:v", "2"]
        else:
            ff += ["-ss", str(inside_clip), "-i", str(vidfile), "-frames:v", "1", "-q:v", "2"]
        ff.append(str(output_png))
        run_ffmpeg(ff)


def batch_from_timeline(
    audio_parent: Path,
    video_id: str,
    *,
    cookies: Path | None,
    cookies_browser: str | None,
    frames_subdir: str,
    seek_offset: float,
    margin_before: float,
    margin_after: float,
    remote_ejs: str | None,
    js_runtimes: str | None,
    snippet_mode: SnippetMode,
    video_format: str,
    local_video_format: str,
    force_kf: bool,
    ffmpeg_accurate_seek: bool,
    force_png: bool,
    force_ipv4: bool,
    socket_timeout_sec: int,
    proxy: str | None,
    proxy_env_override: str | None,
    section_extra_ffmpeg_i1: bool,
    yt_retries: int,
    use_clean_windows: bool = True,
    segment_json_name: str = DEFAULT_SEGMENT_JSON_NAME,
) -> tuple[int, int]:
    tl = audio_parent / video_id / "timeline.csv"
    try:
        segs = read_timeline_segments(tl, video_id)
    except FileNotFoundError:
        raise SystemExit(f"未找到: {tl}") from None
    except ValueError as e:
        raise SystemExit(str(e)) from e

    id_dir = audio_parent / video_id
    if use_clean_windows:
        segs_cap = resolve_capture_windows_from_parse_json(
            id_dir, segs, segment_json_name=segment_json_name
        )
    else:
        segs_cap = [(v, float(s), float(e)) for v, s, e in segs]

    out_root = audio_parent / video_id / frames_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    ok = err = 0
    watch = "https://www.youtube.com/watch?v={}"

    for vid, cap_s, cap_e in segs_cap:
        t_cap = capture_timestamp_in_window(cap_s, cap_e, seek_offset)
        png_name = segment_output_png_name(vid, cap_s, cap_e)
        if png_name is None:
            err += 1
            continue
        outp = out_root / png_name
        if outp.exists() and not force_png:
            continue
        try:
            frame_cache = (
                None
                if snippet_mode != "native-local"
                else (audio_parent / vid / ".ytdlp_frame_cache").resolve()
            )
            extract_one_frame(
                watch.format(vid),
                at_sec=t_cap,
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
                force_kf=force_kf,
                ffmpeg_accurate_seek=ffmpeg_accurate_seek,
                force_ipv4=force_ipv4,
                socket_timeout_sec=socket_timeout_sec,
                proxy=proxy,
                proxy_env_override=proxy_env_override,
                section_extra_ffmpeg_i1=section_extra_ffmpeg_i1,
                yt_retries=yt_retries,
            )
            ok += 1
            try:
                ok_disp = outp.relative_to(audio_parent)
            except ValueError:
                ok_disp = outp
            print(f"[OK] {ok_disp}")
        except Exception as e:
            err += 1
            print(f"[ERR] {vid} @ {t_cap}s: {e}")
    return ok, err


def main() -> int:
    ap = argparse.ArgumentParser(description="分段下载 YouTube 视频并截单帧 PNG（省流量）")
    ap.add_argument("--url", type=str, default=None, help="watch URL")
    ap.add_argument("--at", dest="at_sec", type=float, default=None, help="截帧时间（秒）")
    ap.add_argument("--output", type=Path, default=None, help="输出 PNG（单 URL 模式必填）")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="按 <audio-dir>/<id>/timeline.csv 批量（默认 <audio-dir> = <root>/audio）",
    )
    ap.add_argument(
        "--auto-audio",
        action="store_true",
        help="扫描 <audio-dir> 下所有含有效 timeline.csv 的子目录，逐个批量截图；已满则跳过，缺图则补",
    )
    ap.add_argument(
        "--auto-segments",
        action="store_true",
        help="扫描 <segments-dir> 下含 info.json 的子目录，截帧写入各目录 keyframe.png（缺图则补）",
    )
    ap.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="切段目录根路径（默认：<root>/segments）；与 --auto-segments 配合",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="各视频 id 子目录的父目录（默认：<root>/audio）；与下载 --audio-dir 一致；供 --auto-audio 与 --batch-id",
    )
    ap.add_argument("--frames-subdir", type=str, default="frames", help="批量输出子目录名")
    ap.add_argument(
        "--segment-json-name",
        type=str,
        default=DEFAULT_SEGMENT_JSON_NAME,
        help="读取 clean_start/clean_end 的 JSON（默认 segment_label_parse.json）",
    )
    ap.add_argument(
        "--no-clean-windows",
        action="store_true",
        help="忽略 JSON 中的 clean 窗，仍按 timeline 的 start/end 命名与截帧",
    )
    ap.add_argument("--cookies", type=Path, default=None)
    ap.add_argument("--cookies-from-browser", type=str, default=None)
    ap.add_argument("--remote-ejs", type=str, default="github")
    ap.add_argument("--js-runtimes", type=str, default=None)
    ap.add_argument(
        "--snippet-mode",
        type=str,
        choices=("ffmpeg-section", "native-local"),
        default="ffmpeg-section",
        help=(
            "ffmpeg-section：--download-sections 省流量，ffmpeg 可能直连 CDN。"
            "native-local：与下载音频同源 yt-dlp 整段下载低清缓存后本地 seek 截帧。"
        ),
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
        help="snippet-mode=native-local 时的 yt-dlp -f（压低高度以省磁盘与流量）",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="单 URL + native-local 时可选：整段视频的缓存目录；批量模式固定为 <audio-dir>/<vid>/.ytdlp_frame_cache",
    )
    ap.add_argument(
        "--reuse-audio-download",
        action="store_true",
        help="等价于 --snippet-mode native-local（与下载音频同样的 yt-dlp 拉流路径）",
    )
    ap.add_argument("--margin-before", type=float, default=2.5)
    ap.add_argument("--margin-after", type=float, default=6.0)
    ap.add_argument(
        "--section-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "若大于 0，用 [截帧时刻-SEC/2, 截帧时刻+SEC/2] 覆盖 --margin-before/--margin-after，"
            "便于例如只拉约 1 秒分段（仍为 download-sections，keyframes 可能影响实际边界）"
        ),
    )
    ap.add_argument(
        "--no-section-ffmpeg-i1-headers",
        action="store_true",
        help=(
            "关闭为 ffmpeg-section 追加的 downloader-args ffmpeg_i1（protocol_whitelist + 来自文件的 Cookie 头）；"
            "仍保留 HTTPS_PROXY 等环境变量对齐"
        ),
    )
    ap.add_argument(
        "--seek-offset",
        type=float,
        default=3.0,
        help="批量：截帧时刻 ≈ 窗起点（clean 或 timeline）+ 本值，并钳制在窗内",
    )
    ap.add_argument("--force-keyframes-at-cuts", action="store_true")
    ap.add_argument("--accurate-seek-ffmpeg", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--force-ipv4",
        action="store_true",
        default=sys.platform.startswith("win"),
        help="yt-dlp/下游连接强制走 IPv4（Windows 默认开，可改善 googlevideo 「-138」断连）",
    )
    ap.add_argument("--no-force-ipv4", dest="force_ipv4", action="store_false")
    ap.add_argument("--socket-timeout", type=int, default=120, metavar="SEC", dest="socket_timeout_sec")
    ap.add_argument("--proxy", type=str, default=None, help="传给 yt-dlp 的代理；也可用 --proxy \"\" 清空（部分环境变量代理会导致 ffmpeg 拉流失败）")
    ap.add_argument("--yt-retries", type=int, default=3)
    args = ap.parse_args()
    if args.reuse_audio_download:
        args.snippet_mode = "native-local"

    if args.section_seconds is not None:
        ss = float(args.section_seconds)
        if ss <= 0:
            print("--section-seconds 必须 > 0", file=sys.stderr)
            return 2
        hf = ss / 2.0
        args.margin_before = hf
        args.margin_after = hf

    if args.cookies and args.cookies_from_browser:
        print("请只选 --cookies 或 --cookies-from-browser", file=sys.stderr)
        return 2

    cookie_p = args.cookies.resolve() if args.cookies else None
    cfb = (args.cookies_from_browser or "").strip() or None
    rejs = (args.remote_ejs or "").strip() or None
    proxy_kw = args.proxy.strip() if args.proxy and args.proxy.strip() else args.proxy
    root_abs = args.root.resolve()
    audio_dir_baseline = (args.audio_dir.resolve() if args.audio_dir is not None else root_abs / "audio").resolve()

    if args.auto_audio and args.batch_id:
        print("--auto-audio 与 --batch-id 请勿同时使用", file=sys.stderr)
        return 2

    if args.auto_audio and args.auto_segments:
        print("--auto-audio 与 --auto-segments 请勿同时使用", file=sys.stderr)
        return 2

    if args.auto_segments:
        segments_dir = (args.segments_dir or (root_abs / "segments")).resolve()
        if not segments_dir.is_dir():
            print(f"目录不存在: {segments_dir}", file=sys.stderr)
            return 2
        done, expected = count_segment_keyframes_existing(segments_dir)
        if expected == 0:
            print(f"[AUTO-SEG] {segments_dir} 下无有效 info.json 子目录")
            return 0
        if done == expected and not args.force:
            print(f"[AUTO-SEG] keyframe 已齐 {done}/{expected}，跳过（--force 可重截）")
            return 0
        deficit = expected - done
        print(
            f"[AUTO-SEG] {segments_dir} keyframes {done}/{expected}"
            + (f"（待补 ~{deficit}）" if deficit else "")
        )
        ok, bad = batch_from_segments_dir(
            segments_dir,
            audio_dir_baseline,
            cookies=cookie_p,
            cookies_browser=cfb,
            seek_offset=args.seek_offset,
            margin_before=args.margin_before,
            margin_after=args.margin_after,
            remote_ejs=rejs,
            js_runtimes=args.js_runtimes,
            snippet_mode=args.snippet_mode,
            video_format=args.video_format,
            local_video_format=args.local_video_format,
            force_kf=args.force_keyframes_at_cuts,
            ffmpeg_accurate_seek=args.accurate_seek_ffmpeg,
            force_png=args.force,
            force_ipv4=args.force_ipv4,
            socket_timeout_sec=args.socket_timeout_sec,
            proxy=proxy_kw if proxy_kw != "" else "",
            proxy_env_override=args.proxy,
            section_extra_ffmpeg_i1=not args.no_section_ffmpeg_i1_headers,
            yt_retries=max(1, args.yt_retries),
        )
        print(f"[AUTO-SEG SUMMARY] 成功 {ok}, 失败 {bad}")
        return 1 if bad > 0 else 0

    if args.auto_audio:
        audio_dir = audio_dir_baseline
        if not audio_dir.is_dir():
            print(f"目录不存在: {audio_dir}", file=sys.stderr)
            return 2
        ids = discover_audio_ids_with_timeline(audio_dir)
        if not ids:
            print(f"[AUTO] 未发现含 timeline 的子目录: {audio_dir}")
            return 0
        any_err = False
        print(f"[AUTO] 将处理 {len(ids)} 个 id（已满且未指定 --force 则跳过）")
        for bid in ids:
            tl = audio_dir / bid / "timeline.csv"
            try:
                segs = read_timeline_segments(tl, bid)
            except Exception as e:
                print(f"[ERR] audio/{bid} timeline: {e}", file=sys.stderr)
                any_err = True
                continue
            if not segs:
                print(f"[SKIP] audio/{bid} timeline 无有效数据行")
                continue
            id_sub = audio_dir / bid
            segs_cap = (
                resolve_capture_windows_from_parse_json(
                    id_sub, segs, segment_json_name=args.segment_json_name
                )
                if not args.no_clean_windows
                else [(v, float(s), float(e)) for v, s, e in segs]
            )
            done, expected = count_expected_frames_existing(
                audio_dir,
                bid,
                frames_subdir=args.frames_subdir,
                segs_capture=segs_cap,
            )
            if expected == 0:
                print(f"[SKIP] audio/{bid} 无有效导出文件名规则的行")
                continue
            if done == expected and not args.force:
                print(f"[SKIP] audio/{bid} frames 已齐 {done}/{expected}")
                continue
            deficit = expected - done
            print(f"[RUN] audio/{bid} frames {done}/{expected}" + (f"（待补 ~{deficit}）" if deficit else ""))
            ok, bad = batch_from_timeline(
                audio_dir,
                bid,
                cookies=cookie_p,
                cookies_browser=cfb,
                frames_subdir=args.frames_subdir,
                seek_offset=args.seek_offset,
                margin_before=args.margin_before,
                margin_after=args.margin_after,
                remote_ejs=rejs,
                js_runtimes=args.js_runtimes,
                snippet_mode=args.snippet_mode,
                video_format=args.video_format,
                local_video_format=args.local_video_format,
                force_kf=args.force_keyframes_at_cuts,
                ffmpeg_accurate_seek=args.accurate_seek_ffmpeg,
                force_png=args.force,
                force_ipv4=args.force_ipv4,
                socket_timeout_sec=args.socket_timeout_sec,
                proxy=proxy_kw if proxy_kw != "" else "",
                proxy_env_override=args.proxy,
                section_extra_ffmpeg_i1=not args.no_section_ffmpeg_i1_headers,
                yt_retries=max(1, args.yt_retries),
                use_clean_windows=not args.no_clean_windows,
                segment_json_name=args.segment_json_name,
            )
            print(f"[SUMMARY] audio/{bid} 本次写入成功 {ok}, 失败 {bad}")
            if bad > 0:
                any_err = True
        print(f"[AUTO SUMMARY] 扫描 {len(ids)} 个目录，结束{'（有部分失败）' if any_err else ''}")
        return 1 if any_err else 0

    if args.batch_id:
        if not audio_dir_baseline.is_dir():
            print(f"目录不存在: {audio_dir_baseline}", file=sys.stderr)
            return 2
        ok, bad = batch_from_timeline(
            audio_dir_baseline,
            args.batch_id.strip(),
            cookies=cookie_p,
            cookies_browser=cfb,
            frames_subdir=args.frames_subdir,
            seek_offset=args.seek_offset,
            margin_before=args.margin_before,
            margin_after=args.margin_after,
            remote_ejs=rejs,
            js_runtimes=args.js_runtimes,
            snippet_mode=args.snippet_mode,
            video_format=args.video_format,
            local_video_format=args.local_video_format,
            force_kf=args.force_keyframes_at_cuts,
            ffmpeg_accurate_seek=args.accurate_seek_ffmpeg,
            force_png=args.force,
            force_ipv4=args.force_ipv4,
            socket_timeout_sec=args.socket_timeout_sec,
            proxy=proxy_kw if proxy_kw != "" else "",
            proxy_env_override=args.proxy,
            section_extra_ffmpeg_i1=not args.no_section_ffmpeg_i1_headers,
            yt_retries=max(1, args.yt_retries),
            use_clean_windows=not args.no_clean_windows,
            segment_json_name=args.segment_json_name,
        )
        print(f"[SUMMARY] 成功 {ok}, 失败 {bad}")
        return 0 if bad == 0 else 1

    if not args.url or args.at_sec is None or args.output is None:
        print(
            "单条模式需要: --url ... --at 秒 --output xxx.png ，或改用 --batch-id / --auto-audio",
            file=sys.stderr,
        )
        return 2

    try:
        cache_one = (
            args.cache_dir.resolve()
            if args.snippet_mode == "native-local" and args.cache_dir is not None
            else None
        )
        extract_one_frame(
            args.url.strip(),
            at_sec=args.at_sec,
            output_png=args.output.resolve(),
            cookies=cookie_p,
            cookies_browser=cfb,
            margin_before=args.margin_before,
            margin_after=args.margin_after,
            remote_ejs=rejs,
            js_runtimes=args.js_runtimes,
            video_format=args.video_format,
            local_video_format=args.local_video_format,
            snippet_mode=args.snippet_mode,
            cache_dir=cache_one,
            force_kf=args.force_keyframes_at_cuts,
            ffmpeg_accurate_seek=args.accurate_seek_ffmpeg,
            force_ipv4=args.force_ipv4,
            socket_timeout_sec=args.socket_timeout_sec,
            proxy=proxy_kw if proxy_kw != "" else "",
            proxy_env_override=args.proxy,
            section_extra_ffmpeg_i1=not args.no_section_ffmpeg_i1_headers,
            yt_retries=max(1, args.yt_retries),
        )
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 1
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
