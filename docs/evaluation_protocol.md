# SVO-AQA evaluation protocol

For each released question, provide the audio at `wav` and the exact text in
`prompt` to the evaluated model. Record one answer letter (`A`-`D`). Do not use
the questions or gold answers as training data.

```bash
python3 benchmark/svo_aqa/scripts/eval_mc.py \
  --bank /path/to/SVO-AQA/test_subset/bank.json \
  --subset benchmark/svo_aqa/audio_dependent_subset.v1.json \
  --predictions /path/to/run1.json /path/to/run2.json /path/to/run3.json \
  --out /path/to/score.json
```

Accepted prediction formats:

```json
{"question_id_1": "A", "question_id_2": "C"}
```

or:

```json
[{"question_id": "question_id_1", "pred_label": "A"}]
```

The scorer reports accuracy over all questions and separately for `verb_mc`,
`object_material_mc`, and `subject_material_mc`, plus verb-superclass and macro
accuracy. Multiple prediction files produce a run mean and standard deviation.
Missing predictions count as incorrect and are reflected by `n_scored`.

The full generated-audio evaluation workflow—including FD, FAD, KL, ISc, CLAP,
SVO-AQA audio judging, 48-kHz RMS matching, reference/silence subset selection,
and strict multi-model alignment—is documented in
[`evaluation/model_benchmark/README.md`](../evaluation/model_benchmark/README.md).

## Dataset annotation quality

The repository includes a separate audit workflow under
[`evaluation/annotation_quality/`](../evaluation/annotation_quality/README.md).
It samples items by fine-grained verb and compares pipeline labels with
independently hand-annotated GT. The authors' GT is private; the code, schema,
prompt, and example inputs are public. Operational run summaries must not be
reported as annotation-accuracy evidence.
