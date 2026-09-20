# Reproducibility environment

## Reference environment

DeepASMR-NSpeech v1.0.0 is validated with:

- Ubuntu 24.04.1 LTS (x86_64);
- Python 3.13.5;
- pip 26.2.1;
- FFmpeg 6.1.1;
- the exact Python package versions in `requirements-lock.txt`.

For method development, `requirements.txt` retains compatible lower bounds.
For paper-result reproduction, use the lock file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

Deno is recommended by current `yt-dlp` for some YouTube extraction paths but
is not required by the offline tests. YouTube behavior, source availability,
and browser/cookie requirements can change independently of this repository.

## Released annotation defaults

| Stage | Released default | Temperature | Output limit |
|---|---|---:|---:|
| Train Stage 1: keyframe description | `Qwen/Qwen3.5-397B-A17B` | 0.2 | 1024 in the released Slurm recipe |
| Train Stage 2: class and nouns | `Qwen/Qwen3.6-35B-A3B` | 0.0 | 512 |
| Train Stage 3: audio verb | `qwen3.5-omni-plus` | 0.0 | 320 |
| Annotation-quality noun adjudication | `Qwen/Qwen3.6-35B-A3B` | 0.0 | 256 |

API-backed annotation is not bitwise deterministic. Provider-side model
updates may change results even with fixed prompts and temperature. Retain the
provider, exact model identifier, request settings, date, prompts, response
cache, and run summaries for every reported run.

The frozen v1 alias+LLM audit settings are stored in
`evaluation/annotation_quality/configs/paper_alias_llm_v1.json`.

## Model-evaluation environment

The GPU-heavy FD/FAD/KL/ISc/CLAP environment is isolated from the annotation
environment. It was validated with Python 3.10.20, CUDA 12.1, ffmpeg 6.1.1, and
`requirements-evaluation-lock.txt`. SVO-AQA and global-metric settings are
frozen in `evaluation/model_benchmark/configs/paper_v1.json`; setup and commands
are documented in `evaluation/model_benchmark/README.md`.

## Reproducibility boundary

The repository releases the construction and evaluation methods, frozen noun
alias table, prompt, and model settings. It does not release the authors'
manually annotated quality-control GT or its LLM adjudication cache. Therefore,
third parties can reproduce the protocol on their own human GT, but cannot
independently recompute the exact manual-audit numbers reported in the paper.
