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

**73,829 ten-second clips · 202.6 hours · 37 fine-grained actions · SVO-AQA**

[🎧 Interactive Demo](https://huggingface.co/spaces/yzyai/DeepASMR-NSpeech-Demo) ·
[⌘ Code](https://github.com/bonnie-yzy/DeepASMR-NSpeech) ·
📄 Paper: coming soon

</div>

![DeepASMR-NSpeech verb and material taxonomies](assets/vocab_taxonomy_trees.svg)

DeepASMR-NSpeech is a non-speech ASMR audio dataset with structured
Subject-Verb-Object annotations and an SVO-AQA audio multiple-choice benchmark.

**Authors:** Ziyi Yang, Leying Zhang, Chenda Li, and Yanmin Qian

<sub>Auditory Cognition and Computational Acoustics Lab<br>
Shanghai Jiao Tong University</sub>

**Version:** 1.0.0

## Dataset size

| Metric | Train | Test | Total / cross-split |
|---|---:|---:|---:|
| Source videos | 213 | 297 | 6 shared |
| Creators | 14 | 6 | 4 shared |
| Annotated events | 12,327 | 1,163 | 13,490 |
| Ten-second clips | 72,666 | 1,163 | 73,829; 0 overlap |
| Duration | 199.4 h | 3.19 h | 202.6 h |
| Embedded-audio Parquet | 95.711 GiB | 1.549 GiB | 97.260 GiB |

## Contents

```text
DeepASMR-NSpeech-dataset/
  parquet/train-*.parquet
  parquet/test-*.parquet
  metadata/train.jsonl
  metadata/test.jsonl
  train/train_label.json
  test/test_label.json
  SVO-AQA/test_subset/bank.json
  SVO-AQA/test_subset/audio_dependent_subset.v1.json
  vocab/closed_vocab_reference.md
  release_manifest.json
  parquet_manifest.json
```

The Parquet shards contain embedded audio bytes and the public fields
`audio_id`, `audio`, `caption`, `verb`, `subject`, `object`, and `creator_id`.
This follows the public DeepASMR release pattern and avoids publishing
machine-specific source paths.

```python
from datasets import load_dataset

dataset = load_dataset("yzyai/DeepASMR-NSpeech")
sample = dataset["train"][0]
print(sample["caption"], sample["audio"])
```

## Demo

The [interactive project demo](https://huggingface.co/spaces/yzyai/DeepASMR-NSpeech-Demo)
contains reference recordings, representative model generations, keyframes,
the full verb/material taxonomy, and worked SVO-AQA examples. Headphones are
recommended for the low-intensity ASMR recordings.

Clip-label files contain relative `wav`, `caption`, closed-set `verb`, open noun
phrases `subject` and `object`, and anonymized `creator_id`. Raw creator timeline
text, titles, channel names, API metadata, and private quality-control GT are not
part of the public label files.

## Creation

The public code repository provides the complete collection and annotation
path: timeline acquisition, normalization/translation, segmentation, keyframe
extraction, multimodal annotation, vocabulary normalization, label export, and
SVO-AQA construction. SVO-AQA starts from human base annotations and can be
extended by adding rows following the documented schema.

The repository also provides annotation-quality audit code. Its GT must be
independently labeled by humans; the authors' private GT is not distributed.

## Evaluation

For SVO-AQA, send each question's audio and prompt to a model and compare its
single-letter answer with `gold_label`. Report overall and per-task accuracy.
The full bank contains 431 human-validated questions. The paper's main table
uses the frozen 55-question audio-dependent subset distributed beside the bank;
the GitHub repository provides the judge, subset-selection, and multi-run
scoring code.

For dataset-label quality, report strict metrics against manual GT separately
from any alias- or LLM-assisted entity-resolution score. Processing success
counts are operational statistics, not label-quality metrics.

## License, limitations, and source-media rights

The structured annotations, dataset metadata, and SVO-AQA benchmark annotations
that the authors are authorized to license are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Structured
labels may contain model-assisted annotation noise.

The source media originates from third-party public videos. All copyright and
associated rights in the original source videos, original audio, and
creator-produced content remain with their respective original creators or
other rights holders. CC BY-NC 4.0 does not transfer ownership of that content
or grant rights that the dataset authors do not hold. Users must comply with
applicable creator rights, platform terms, privacy requirements, and local law.

## Links

- Demo: https://huggingface.co/spaces/yzyai/DeepASMR-NSpeech-Demo
- Project page: https://bonnie-yzy.github.io/DeepASMR-NSpeech/
- Code: https://github.com/bonnie-yzy/DeepASMR-NSpeech
- Dataset: https://huggingface.co/datasets/yzyai/DeepASMR-NSpeech
- Paper: `TBD`

Replace the paper URL after publication, and keep the paper and repository
rights statements aligned.
