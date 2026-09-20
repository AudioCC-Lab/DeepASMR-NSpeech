# Test-set model evaluation

This directory evaluates generated audio from any text-to-audio model on the
same DeepASMR-NSpeech test IDs. It contains the two protocols reported in the
paper:

1. global FD, FAD, KL, Inception Score (ISc), and CLAP metrics; and
2. SVO-AQA verb, verb-superclass, subject-material, object-material, and macro
   accuracy.

Model generation code is not required. Each model supplies one WAV named
`{audio_id}.wav` for every evaluated test item. The reference labels and audio
come from the released Hugging Face dataset.

## 1. Prepare aligned manifests for multiple models

Copy and edit `examples/models.example.json`, or use
`examples/models.paper.example.json` for the six paper systems (AudioLDM,
AudioLDM2, Tango2, Stable Audio Open, Stable Audio 3, and UniFlow-Audio). The
relative paths are examples and may be changed in an untracked local copy.

```bash
python3 evaluation/model_benchmark/prepare_manifests.py \
  --labels DeepASMR-NSpeech-dataset/test/test_label.json \
  --dataset-root DeepASMR-NSpeech-dataset \
  --models work/models.json \
  --out-dir work/eval_manifests
```

Strict coverage is the default. The command fails if a reference or model
output is missing so different models cannot silently use different subsets.
`--allow-missing` is provided only for diagnostics; report the resulting
intersection size if it is ever used.

## 2. Global metrics

The paper environment used Python 3.10.20, ffmpeg 6.1.1, the versions in
`requirements-evaluation-lock.txt`, and the evaluation code shipped with
UniFlow-Audio commit `61263340a479da4da875664af48cd7445d70531f`. The included
patch freezes the memory, cache, resampling, and device compatibility changes
used on the cluster.

```bash
python3.10 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-evaluation-lock.txt

git clone https://github.com/wsntxxn/UniFlow-Audio.git third_party/UniFlow-Audio
git -C third_party/UniFlow-Audio checkout 61263340a479da4da875664af48cd7445d70531f
git -C third_party/UniFlow-Audio apply \
  "$PWD/evaluation/model_benchmark/patches/uniflow_audio_eval_612633.patch"
export PYTHONPATH="$PWD/third_party/UniFlow-Audio/evaluation:$PWD/third_party/UniFlow-Audio:${PYTHONPATH:-}"
```

Evaluate each model manifest:

```bash
python3 evaluation/model_benchmark/global_metrics.py \
  --reference-audio work/eval_manifests/reference_audio.jsonl \
  --reference-captions work/eval_manifests/reference_captions.jsonl \
  --generated-audio work/eval_manifests/audioldm.jsonl \
  --model-id audioldm \
  --out results/audioldm_global.json \
  --recalculate
```

The default LAION-CLAP checkpoint is `630k-audioset-best.pt`; pass
`--clap-checkpoint` or set `CLAP_MODEL_PATH` to an already downloaded copy.
The output's `paper_metrics` field contains the five table values.

## 3. SVO-AQA model judging

The base SVO-AQA annotations are human-created. The public bank contains 431
questions, while the paper's main table uses the frozen 55-question
audio-dependent subset in
`benchmark/svo_aqa/audio_dependent_subset.v1.json`.

Keep the API key in the shell, not in a file committed to Git:

```bash
export DASHSCOPE_API_KEY="..."

for run in 1 2 3; do
  python3 evaluation/model_benchmark/run_svo_aqa_judge.py \
    --bank DeepASMR-NSpeech-dataset/SVO-AQA/test_subset/bank.json \
    --subset benchmark/svo_aqa/audio_dependent_subset.v1.json \
    --audio-dir work/generated/model \
    --reference-audio-dir DeepASMR-NSpeech-dataset/test/audio \
    --model qwen3.5-omni-flash \
    --out "results/model_svo_aqa_run${run}.json"
done
```

Generated clips are resampled to 48 kHz, converted to mono 16-bit PCM,
RMS-matched to the corresponding reference, and peak-limited at 0.99. The
judge uses temperature 0, at most 8 output tokens, top-5 log probabilities, and
the highest first non-empty-token log probability over the answer letters.

Average and score the three runs:

```bash
python3 benchmark/svo_aqa/scripts/eval_mc.py \
  --bank DeepASMR-NSpeech-dataset/SVO-AQA/test_subset/bank.json \
  --subset benchmark/svo_aqa/audio_dependent_subset.v1.json \
  --predictions results/model_svo_aqa_run1.json \
                results/model_svo_aqa_run2.json \
                results/model_svo_aqa_run3.json \
  --out results/model_svo_aqa_summary.json
```

To rebuild the 55-question subset from three reference and three silence
control runs, use `select_audio_dependent_subset.py`. Its default rule is
reference accuracy at least 2/3 and silence accuracy at most 1/3 per question.
The paper used `qwen3.5-omni-plus-2026-03-15` for this selection step and
`qwen3.5-omni-flash` for evaluating generated systems; the distinction is
frozen in `configs/paper_v1.json`.

Hosted model aliases and backend implementations may change. The frozen code,
prompt, IDs, and parameters reproduce the evaluation method; exact future API
scores additionally depend on provider access to the recorded model version.
