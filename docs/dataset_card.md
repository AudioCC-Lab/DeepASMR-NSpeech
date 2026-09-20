---
pretty_name: DeepASMR-NSpeech
version: 1.0.0
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-to-audio
- audio-classification
tags:
- audio
- asmr
- sound-generation
- subject-verb-object
- benchmark
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: "parquet/train-*.parquet"
  - split: test
    path: "parquet/test-*.parquet"
---

<div align="center">

# DeepASMR-NSpeech

### A Fine-Grained Benchmark for Non-Speech ASMR Generation

**73,829 ten-second clips · 202.6 hours · 37 fine-grained actions**

[🎧 Interactive Demo](https://huggingface.co/spaces/AudioCC-Lab/DeepASMR-NSpeech-Demo) ·
[⌘ Code](https://github.com/AudioCC-Lab/DeepASMR-NSpeech) ·
📄 Paper: coming soon

</div>

## Dataset summary

DeepASMR-NSpeech is a non-speech ASMR audio dataset with structured Subject-Verb-Object annotations and the SVO-AQA audio question-answering benchmark. All clips are 10 seconds long and cover 37 fine-grained sound-producing actions.

**Authors:** Ziyi Yang, Leying Zhang, Chenda Li, and Yanmin Qian

<sub>Auditory Cognition and Computational Acoustics Lab<br>
Shanghai Jiao Tong University</sub>

## Dataset statistics

| Metric | Train | Test | Total / cross-split |
|---|---:|---:|---:|
| Annotated events | 12,327 | 1,163 | 13,490 |
| Ten-second clips | 72,666 | 1,163 | 73,829; 0 overlap |
| Duration | 199.4 h | 3.19 h | 202.6 h |

## Quick start

```python
from datasets import load_dataset

dataset = load_dataset("AudioCC-Lab/DeepASMR-NSpeech")
sample = dataset["train"][0]
print(sample["caption"], sample["audio"])
```

## Data fields

Each Parquet row contains `audio_id`, embedded `audio`, `caption`, closed-set `verb`, open noun phrases `subject` and `object`, and anonymized `creator_id`.

The release also includes 431 human-validated SVO-AQA questions and a frozen 55-question audio-dependent subset. See the [code repository](https://github.com/AudioCC-Lab/DeepASMR-NSpeech) for construction and evaluation instructions.

## License and source-media rights

The structured annotations, dataset metadata, and SVO-AQA annotations that the authors are authorized to license are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The source media originates from third-party public videos. All associated rights remain with their respective original creators or other rights holders. Users must comply with applicable creator rights, platform terms, privacy requirements, and local law. Structured labels may contain model-assisted annotation noise.

## Citation

```bibtex
@misc{yang2026deepasmrnspeech,
  title  = {DeepASMR-NSpeech: A Fine-Grained Benchmark for Non-Speech ASMR Generation},
  author = {Yang, Ziyi and Zhang, Leying and Li, Chenda and Qian, Yanmin},
  year   = {2026},
  note   = {Dataset and code, version 1.0.0}
}
```
