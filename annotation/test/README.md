# Test annotation path

The test path drafts SVO labels from segment text and visual context, then
normalizes them to the released taxonomy. It does not compare against private
manual GT.

```bash
python3 annotation/test/init_test_v5.py \
  --context-json /path/to/test_context.json \
  --out-json annotation/test/test_v5.json

python3 annotation/test/run_seg_label_svo.py \
  --input-json annotation/test/test_v5.json \
  --continue-on-error

python3 annotation/test/run_svo_normalize.py \
  --input annotation/test/outputs/seg_label_svo.json \
  --continue-on-error

python3 annotation/test/build_test_label.py \
  --input annotation/test/outputs/seg_label_svo_normalized.json \
  --output test_label.json
```

Prompts are in `annotation/test/prompts/`. Missing inputs, API failures, and
normalization failures are included in adjacent `*.run_summary.json` files.
Rows requiring human completion may carry `needs_manual_review`; this is a
workflow flag, not an accuracy score.
