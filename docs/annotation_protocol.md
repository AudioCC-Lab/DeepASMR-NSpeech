# Annotation protocol

## Evidence flow

```text
creator timeline label -> temporal segment -> representative keyframe
  -> visual description -> text SVO draft -> audio verb selection
  -> vocabulary normalization -> public clip label
```

Timeline text is a weak human-provided cue. The visual stage identifies the
foreground interaction; the text stage proposes subject/object and a verb
superclass; the audio stage resolves the fine-grained closed-set verb.

## Train stages

| Stage | Inputs | Outputs |
|---|---|---|
| Vision | event + keyframe | `visual_description`, model/runtime metadata |
| Text | segment label + visual description | verb-class shortlist, subject, object |
| Audio | event audio + text context | closed-set `verb`, caption |
| Export | annotated events + clip paths | `train_label.json` |

## Test stages

The test pipeline initializes rows from prepared segments, extracts raw SVO,
normalizes verbs/nouns, and exports public labels. `fail` means a field could
not be inferred; `null` is valid for a missing second interaction object.

## Vocabulary policy

- Fine verbs: 37 verbs grouped under 18 superclasses.
- Materials: 12 acoustic material superclasses used by SVO-AQA.
- `vocab/stage2_class_disambiguation.txt` defines within-superclass boundaries.
- Alias CSVs normalize spelling and phrasing; they do not replace human review.

All prompt templates used by these stages are included under `annotation/`.

## Operational statistics

Each stage reports counts and execution failures. Typical skip reasons include
missing images/audio/context, already-completed rows, or filtered inputs. API
errors are classified into network, timeout, rate-limit, authentication, and
generic API categories. These statistics answer “how many items completed and
why did processing fail?” They do not compare labels with private ground truth.

## Public/private boundary

Public: collection, prompts, taxonomies, annotation stages, normalized public
labels, annotation-quality audit code, and SVO-AQA construction/evaluation.
Private: the authors' hand-labeled quality-control GT, reports/caches that reveal
that GT, and generated-audio SVO quality metrics. Anyone rerunning the audit must
create an independent manual GT; pipeline output is not valid ground truth.
