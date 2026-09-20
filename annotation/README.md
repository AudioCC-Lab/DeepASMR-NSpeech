# Annotation pipelines

DeepASMR-NSpeech uses multimodal annotation stages with public prompts and
closed vocabularies. The public annotation-quality evaluator is provided under
[`../evaluation/annotation_quality/`](../evaluation/annotation_quality/README.md);
the authors' private hand-labeled GT is not included.

## Train input schema

The three train stages share one event-level working JSON. A minimal source row
has the following structure:

```json
{
  "events": [
    {
      "id": "video_or_event_id",
      "start_sec": 120,
      "end_sec": 180,
      "label": "glass tapping",
      "title": "optional source title",
      "author": "optional source creator",
      "segments": [
        {"audio_path": "data/audio/example_000.wav"}
      ]
    }
  ]
}
```

Stage 1 expects one representative PNG under `--figs-dir`, named
`{id}_{int(start_sec)}_{int(end_sec)}.png`. Segment `audio_path` values are read
by Stage 3 and should be valid from the repository working directory or be
absolute runtime inputs. Do not commit machine-specific absolute paths.

## Train split

1. `stage1/run_stage1_vision.py`: keyframe to visual event description.
2. `stage2/run_stage2_text_class_noun.py`: timeline label + visual context to
   verb superclass shortlist, subject, and object.
3. `stage3/run_stage3_audio_verb.py`: audio + context to one closed-set verb.
4. `export_train_label.py`: expand event annotations to clip rows.

All three stages update the same working JSON. Use `--dry-run` and `--limit`
before a full API run. Each stage supports checkpointing and writes its own
operational `run_summary.json`.

## Complete three-stage train commands

Run from the repository root. The examples keep generated files under `work/`,
which is not part of the release:

```bash
export TRAIN_EVENTS_JSON=data/train_events.json
export TRAIN_FRAMES_DIR=data/train_frames
export SILICONFLOW_API_KEY="..."
export DASHSCOPE_API_KEY="..."

mkdir -p work/train_annotation

# Stage 1: representative frame -> visual description
python3 annotation/train/stage1/run_stage1_vision.py \
  --source-json "$TRAIN_EVENTS_JSON" \
  --out-json work/train_annotation/train_events.json \
  --figs-dir "$TRAIN_FRAMES_DIR" \
  --no-enable-thinking \
  --max-tokens 1024 \
  --continue-on-error

# Stage 2: timeline text + visual description -> verb class, subject, object
python3 annotation/train/stage2/run_stage2_text_class_noun.py \
  --out-json work/train_annotation/train_events.json \
  --temperature 0 \
  --max-tokens 512 \
  --continue-on-error

# Stage 3: audio + Stage-2 context -> one closed-set fine verb
python3 annotation/train/stage3/run_stage3_audio_verb.py \
  --out-json work/train_annotation/train_events.json \
  --temperature 0 \
  --max-tokens 320 \
  --continue-on-error

# Event annotations -> clip-level public labels
python3 annotation/train/export_train_label.py \
  --src-json work/train_annotation/train_events.json \
  --out-json work/train_annotation/train_label.json
```

For a three-event smoke run, add `--limit 3` to Stages 1-3. Add `--dry-run`
to one stage to inspect its candidate selection without making API calls. A
dry-run does not fabricate outputs required by the next stage, so a complete
end-to-end result requires real Stage-1, Stage-2, and Stage-3 calls.

Slurm entry points are included alongside each stage. They use repository- and
environment-relative paths rather than cluster-specific absolute paths.

## Test split

1. `test/init_test_v5.py`
2. `test/run_seg_label_svo.py`
3. `test/run_svo_normalize.py`
4. `test/build_test_label.py`

See [`test/README.md`](test/README.md) for commands.

## Configuration

- SiliconFlow/OpenAI-compatible stages read `SILICONFLOW_API_KEY` or
  `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.
- The audio stage reads `DASHSCOPE_API_KEY` and optional
  `DASHSCOPE_BASE_URL`.
- Example env files contain empty values only.
- Vocabulary and disambiguation files are read from the repository-level
  `vocab/` directory.
- Released model and decoding defaults are frozen in
  [`../docs/reproducibility.md`](../docs/reproducibility.md).
