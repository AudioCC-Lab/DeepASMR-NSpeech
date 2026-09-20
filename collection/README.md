# Data collection and construction

This directory contains the complete pre-annotation path used by
DeepASMR-NSpeech. Use only content you are permitted to access and redistribute,
and comply with platform terms and creator rights.

## Main sequence

| Step | Script | Result |
|---:|---|---|
| 1 | `download_from_channel_videos.py` | Enumerate a channel and optionally download each public video |
| 2 | `download_yt_audio_and_timeline.py` | Audio, metadata, and creator-provided timeline CSV |
| 3 | `fetch_train_timelines.py` | Fetch/cache timelines for an existing train split |
| 4 | `normalize_timeline_csv.py` | Normalize timestamp and segment-title columns |
| 5 | `translate_timeline_segment_titles.py` | Translate non-English segment labels while retaining originals |
| 6 | `clean_audio_cut.py` / `clean_audio_cut_wrap.py` | Refine event boundaries from audio |
| 7 | `export_clean_audio_segments.py` | Export segment-level clips |
| 8 | `extract_train_event_frames.py` | Extract one representative frame per train event |
| 9 | `extract_info_full_keyframes.py` / `yt_frame_extract.py` | Extract keyframes for label records or timeline segments |
| 10 | `sort_train_json.py`, `aggregate_verb_object_subject.py` | Deterministic ordering and label inventory helpers |

Supporting tools cover duplicate IDs, missing timelines, preview removal, event
split statistics, cookie filtering, and segment layout.

## Authentication

Cookies are optional runtime inputs:

```bash
python3 collection/download_yt_audio_and_timeline.py \
  --url-file /path/to/urls.txt \
  --root /path/to/data \
  --cookies /secure/path/youtube-cookies.txt \
  --js-runtimes deno \
  --continue-on-error
```

No cookie file is shipped. Browser-cookie flags are also supported by the
download and frame scripts. Do not copy real cookies into this repository.

## Existing-split timeline and keyframe path

```bash
python3 collection/fetch_train_timelines.py \
  --root /path/to/data \
  --train-json /path/to/train.json \
  --continue-on-error

python3 collection/extract_train_event_frames.py \
  --root /path/to/data \
  --train-events /path/to/train_events.json \
  --continue-on-error
```

For timeline translation:

```bash
python3 collection/translate_timeline_segment_titles.py --root /path/to/data
```

## Operational reports

Network and media stages write JSON summaries such as:

- `download_run_summary.json`
- `channel_download_run_summary.json`
- `fetch_run_summary.json`
- `translation_run_summary.json`
- `extract_run_summary.json`
- `keyframe_run_summary.json`

Failures are grouped into reasons such as `network`, `timeout`, `rate_limit`,
`source_unavailable`, `source_too_long`, `media_processing`, and `missing_input`.
These reports measure pipeline completion, not label correctness.
