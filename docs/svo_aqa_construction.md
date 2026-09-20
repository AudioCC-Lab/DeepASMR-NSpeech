# SVO-AQA construction

SVO-AQA is an audio multiple-choice benchmark for fine-grained ASMR verbs,
subject materials, and object materials.

## Human base annotations

The starting `material.json` and `verb.json` files are manually annotated.
They are not outputs of the question builder. Each row contains the clip ID,
relative WAV path, SVO context, relevant material/verb fields, and a human
`same` flag indicating whether the audible event agrees with the annotation.

Minimal fields used across the two files:

```json
{
  "audio_id": "clip_stem",
  "wav": "test/audio/clip_stem.wav",
  "caption": "fingers tapping glass",
  "verb": "tapping",
  "verb_class": "impulsive_contact",
  "subject_core": "fingers",
  "object_core": "surface",
  "subject_material": "textile_fibrous",
  "object_material": "glass",
  "same": true
}
```

Add new rows or fields for another dataset while retaining stable `audio_id`
values and the vocabulary names in `vocab/`. This is the supported extension
point; private annotation-quality GT is not required by the builder.

## Build

```bash
python3 benchmark/svo_aqa/scripts/build_aqascore_clusters.py \
  --material-json /path/to/material.json \
  --verb-json /path/to/verb.json \
  --out-dir build/svo_aqa

python3 benchmark/svo_aqa/scripts/build_mc_same_true_bank.py \
  --annotations-dir build/svo_aqa \
  --material-json /path/to/material.json \
  --verb-json /path/to/verb.json \
  --out-dir build/svo_aqa/mc_same_true
```

The first script creates strict minimal-pair clusters and curated single-clip
questions. Material distractors avoid predefined acoustically near families;
verb padding prefers different verb superclasses. The second script keeps
human `same:true` clips, splits tiers, and writes machine JSON plus review CSV.

## Released evaluator

`benchmark/svo_aqa/scripts/eval_mc.py` accepts model-produced answer letters
and reports overall and per-task accuracy. No model-specific inference routes,
generated-audio quality metrics, or private GT evaluators are included.
