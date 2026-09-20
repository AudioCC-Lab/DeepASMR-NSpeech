from __future__ import annotations

import json
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.test.build_test_label import export_rows
from deepasmr_nspeech import RunSummary, classify_failure
from evaluation.model_benchmark.prepare_manifests import prepare


class ReleasePipelineTests(unittest.TestCase):
    def test_release_uses_mit_code_and_cc_by_nc_data_licenses(self) -> None:
        self.assertTrue((ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"))
        card = (ROOT / "docs/dataset_card.md").read_text(encoding="utf-8")
        self.assertIn("license: cc-by-nc-4.0", card.split("---", 2)[1])

    def test_frozen_annotation_quality_configuration(self) -> None:
        config_path = ROOT / "evaluation/annotation_quality/configs/paper_alias_llm_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_dir = config_path.parent
        for path_key, hash_key in (
            ("noun_alias_csv", "noun_alias_sha256"),
            ("prompt", "prompt_sha256"),
        ):
            source = (config_dir / config[path_key]).resolve()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(digest, config[hash_key])
        alias_lines = (config_dir / config["noun_alias_csv"]).resolve().read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(alias_lines) - 1, 436)
        self.assertEqual(config["model"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(config["temperature"], 0.0)
        self.assertEqual(config["max_tokens"], 256)

    def test_frozen_model_evaluation_configuration(self) -> None:
        config_path = ROOT / "evaluation/model_benchmark/configs/paper_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        patch_path = (config_path.parent / config["global_metrics"]["compatibility_patch"]).resolve()
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        self.assertEqual(digest, config["global_metrics"]["compatibility_patch_sha256"])
        self.assertEqual(config["svo_aqa"]["subset_selection_model"], "qwen3.5-omni-plus-2026-03-15")
        self.assertEqual(config["svo_aqa"]["evaluation_model"], "qwen3.5-omni-flash")
        subset = json.loads((ROOT / "benchmark/svo_aqa/audio_dependent_subset.v1.json").read_text())
        self.assertEqual(subset["meta"]["n_questions"], 55)
        self.assertEqual(subset["meta"]["n_clips"], 50)
        self.assertEqual(len(subset["questions"]), 55)

    def test_run_summary_classifies_and_bounds_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.json"
            summary = RunSummary("test", max_failure_details=1)
            summary.add("success", item_id="ok")
            summary.add("failed", item_id="a", detail="HTTP 429 too many requests")
            summary.add("failed", item_id="b", detail="ffmpeg decode failed")
            payload = summary.finish(out)
            self.assertEqual(payload["successful"], 1)
            self.assertEqual(payload["failed"], 2)
            self.assertEqual(payload["failure_reasons"]["rate_limit"], 1)
            self.assertEqual(payload["failure_reasons"]["media_processing"], 1)
            self.assertEqual(len(payload["failures"]), 1)
            self.assertEqual(payload["dropped_failure_details"], 1)
            self.assertEqual(classify_failure("video too long"), "source_too_long")

    def test_public_test_export_filters_failures_and_duplicates(self) -> None:
        rows = [
            {
                "wav": "test/audio/a.wav",
                "caption": "fingers tapping glass",
                "verb": "tapping",
                "subject": "fingers",
                "object": "glass",
                "seg_label": "glass tapping",
                "status": "ok",
            },
            {
                "wav": "test/audio/b.wav",
                "caption": "duplicate",
                "verb": "tapping",
                "subject": "fingers",
                "object": "glass",
                "seg_label": "glass tapping",
                "status": "ok",
            },
            {"wav": "test/audio/c.wav", "verb": None, "status": "fail"},
        ]
        exported, stats = export_rows(
            rows,
            verb_classes=ROOT / "vocab" / "asmr_verb_classes_en.csv",
            include_failed=False,
            include_trace_fields=False,
        )
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["verb_class"], "impulsive_contact")
        self.assertNotIn("seg_label", exported[0])
        self.assertEqual(stats["skip_reasons"]["duplicate_svo_and_segment_label"], 1)
        self.assertEqual(stats["skip_reasons"]["status_not_ok"], 1)

    def test_svo_aqa_build_and_score(self) -> None:
        base_rows = [
            {
                "audio_id": "clip_a",
                "wav": "test/audio/clip_a.wav",
                "caption": "fingers tapping glass",
                "verb": "tapping",
                "verb_class": "impulsive_contact",
                "subject": "fingers",
                "object": "surface",
                "subject_core": "fingers",
                "object_core": "surface",
                "subject_material": "textile_fibrous",
                "object_material": "glass",
                "same": True,
            },
            {
                "audio_id": "clip_b",
                "wav": "test/audio/clip_b.wav",
                "caption": "fingers tapping wood",
                "verb": "scratching",
                "verb_class": "friction_contact",
                "subject": "fingers",
                "object": "surface",
                "subject_core": "fingers",
                "object_core": "surface",
                "subject_material": "textile_fibrous",
                "object_material": "wood",
                "same": True,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "material.json").write_text(json.dumps(base_rows), encoding="utf-8")
            (work / "verb.json").write_text(json.dumps(base_rows), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark/svo_aqa/scripts/build_aqascore_clusters.py"),
                    "--material-json",
                    str(work / "material.json"),
                    "--verb-json",
                    str(work / "verb.json"),
                    "--out-dir",
                    str(work),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark/svo_aqa/scripts/build_mc_same_true_bank.py"),
                    "--annotations-dir",
                    str(work),
                    "--out-dir",
                    str(work / "bank"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            bank_path = work / "bank/tier1/bank.json"
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            self.assertGreater(len(bank["questions"]), 0)
            predictions = {q["question_id"]: q["gold_label"] for q in bank["questions"]}
            prediction_path = work / "predictions.json"
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            score_path = work / "score.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark/svo_aqa/scripts/eval_mc.py"),
                    "--bank",
                    str(bank_path),
                    "--predictions",
                    str(prediction_path),
                    "--out",
                    str(score_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            score = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertEqual(score["accuracy"], 1.0)

    def test_svo_aqa_multirun_subset_and_superclass_scoring(self) -> None:
        questions = [
            {
                "question_id": "verb_1",
                "audio_id": "a",
                "task": "verb_mc",
                "gold_label": "A",
                "options": [
                    {"label": "A", "value": "tapping", "verb_class": "impact"},
                    {"label": "B", "value": "patting", "verb_class": "impact"},
                    {"label": "C", "value": "rubbing", "verb_class": "friction"},
                ],
            },
            {
                "question_id": "subject_1",
                "audio_id": "b",
                "task": "subject_material_mc",
                "gold_label": "B",
                "options": [{"label": "A", "value": "wood"}, {"label": "B", "value": "skin"}],
            },
            {
                "question_id": "object_1",
                "audio_id": "c",
                "task": "object_material_mc",
                "gold_label": "A",
                "options": [{"label": "A", "value": "glass"}, {"label": "B", "value": "wood"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            bank = work / "bank.json"
            subset = work / "subset.json"
            bank.write_text(json.dumps({"questions": questions}), encoding="utf-8")
            subset.write_text(
                json.dumps({"questions": [{"question_id": row["question_id"]} for row in questions]}),
                encoding="utf-8",
            )
            prediction_paths = []
            for index, predictions in enumerate(
                (
                    {"verb_1": "A", "subject_1": "B", "object_1": "A"},
                    {"verb_1": "B", "subject_1": "B", "object_1": "A"},
                    {"verb_1": "C", "subject_1": "A", "object_1": "A"},
                ),
                start=1,
            ):
                path = work / f"run{index}.json"
                path.write_text(json.dumps(predictions), encoding="utf-8")
                prediction_paths.append(path)
            out = work / "score.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmark/svo_aqa/scripts/eval_mc.py"),
                    "--bank",
                    str(bank),
                    "--subset",
                    str(subset),
                    "--predictions",
                    *(str(path) for path in prediction_paths),
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            score = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(score["n_runs"], 3)
            self.assertAlmostEqual(score["per_task"]["verb_mc"]["accuracy"], 1 / 3)
            self.assertAlmostEqual(score["verb_superclass"]["accuracy"], 2 / 3)
            self.assertAlmostEqual(score["per_task"]["object_material_mc"]["accuracy"], 1.0)

    def test_model_manifests_require_common_test_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            dataset = work / "dataset"
            reference_dir = dataset / "test/audio"
            model_dir = work / "model_outputs"
            reference_dir.mkdir(parents=True)
            model_dir.mkdir()
            rows = []
            for audio_id in ("clip_a", "clip_b"):
                (reference_dir / f"{audio_id}.wav").write_bytes(b"RIFF")
                (model_dir / f"{audio_id}.wav").write_bytes(b"RIFF")
                rows.append(
                    {
                        "audio_id": audio_id,
                        "wav": f"test/audio/{audio_id}.wav",
                        "caption": f"caption {audio_id}",
                    }
                )
            labels = work / "test_label.json"
            models = work / "models.json"
            labels.write_text(json.dumps({"data": rows}), encoding="utf-8")
            models.write_text(
                json.dumps({"models": [{"id": "model_a", "audio_dir": str(model_dir)}]}),
                encoding="utf-8",
            )
            out_dir = work / "manifests"
            report = prepare(
                labels=labels,
                dataset_root=dataset,
                models_json=models,
                out_dir=out_dir,
            )
            self.assertEqual(report["n_aligned"], 2)
            self.assertEqual(len((out_dir / "model_a.jsonl").read_text().splitlines()), 2)
            (model_dir / "clip_b.wav").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(
                    labels=labels,
                    dataset_root=dataset,
                    models_json=models,
                    out_dir=work / "strict_failure",
                )

    def test_hugging_face_staging_removes_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            label_path = work / "train_label.json"
            label_path.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "wav": str(work / "missing.wav"),
                                "caption": "fingers tapping glass",
                                "verb": "tapping",
                                "subject": "fingers",
                                "object": "glass",
                                "author": "private channel name",
                                "seg_label": "private timeline cue",
                                "title": "private title",
                                "subject_material": "skin",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = work / "release"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_deepasmr_nspeech_hf.py"),
                    "--labels-only",
                    "--split",
                    "train",
                    "--train-label",
                    str(label_path),
                    "--out-root",
                    str(out),
                    "--skip-svo-aqa",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            released = json.loads((out / "train/train_label.json").read_text(encoding="utf-8"))
            row = released["data"][0]
            self.assertEqual(row["creator_id"], "creator_01")
            for private_key in ("author", "seg_label", "title", "subject_material"):
                self.assertNotIn(private_key, row)
            manifest = json.loads((out / "release_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "DeepASMR-NSpeech")
            self.assertNotIn(str(work), json.dumps(manifest))
            self.assertIn("license: cc-by-nc-4.0", (out / "README.md").read_text(encoding="utf-8"))

    @unittest.skipUnless(
        importlib.util.find_spec("datasets") and importlib.util.find_spec("pyarrow"),
        "datasets and pyarrow are optional release dependencies",
    )
    def test_hugging_face_parquet_embeds_audio_without_private_paths(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows = {}
            for split in ("train", "test"):
                wav = work / f"{split}.wav"
                with wave.open(str(wav), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(b"\x00\x00" * 160)
                rows[split] = {
                    "wav": str(wav),
                    "caption": "fingers tapping glass",
                    "verb": "tapping",
                    "subject": "fingers",
                    "object": "glass",
                    "author": "private creator name",
                    "title": "private title",
                    "seg_label": "private timeline cue",
                }
                (work / f"{split}.json").write_text(
                    json.dumps({"data": [rows[split]]}), encoding="utf-8"
                )
            out = work / "release"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_hf_parquet.py"),
                    "--train-label",
                    str(work / "train.json"),
                    "--test-label",
                    str(work / "test.json"),
                    "--out-root",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            train_path = next((out / "parquet").glob("train-*.parquet"))
            table = pq.read_table(train_path)
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(
                set(table.column_names),
                {"audio_id", "audio", "caption", "verb", "subject", "object", "creator_id"},
            )
            audio = table["audio"].to_pylist()[0]
            self.assertTrue(audio["bytes"].startswith(b"RIFF"))
            self.assertEqual(audio["path"], "train.wav")
            self.assertNotIn(str(work), json.dumps(table.to_pylist(), default=str))
            metadata = (out / "metadata/train.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(str(work), metadata)
            self.assertNotIn("private creator name", metadata)

    def test_manual_gt_sampling_is_blind_and_stratified(self) -> None:
        rows = [
            {"wav": f"test/audio/tap_{i}.wav", "subject": "nails", "verb": "tapping", "object": "glass"}
            for i in range(4)
        ] + [
            {"wav": f"test/audio/rub_{i}.wav", "subject": "hands", "verb": "rubbing", "object": "cloth"}
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            predictions = work / "predictions.json"
            predictions.write_text(json.dumps({"data": rows}), encoding="utf-8")
            output = work / "manual_gt.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "evaluation/annotation_quality/sample_human_gt.py"),
                    "--predictions",
                    str(predictions),
                    "--sample-size",
                    "4",
                    "--seed",
                    "7",
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            sampled = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(sampled["data"]), 4)
            self.assertEqual(sampled["meta"]["selected_stratum_counts"], {"rubbing": 2, "tapping": 2})
            for row in sampled["data"]:
                self.assertEqual(row["gt"], {"subject": None, "verb": None, "object": None})
                self.assertNotIn("prediction", row)
                self.assertNotIn("verb", row)

    def test_annotation_quality_uses_manual_gt_and_alias_rules(self) -> None:
        predictions = {
            "data": [
                {
                    "wav": "test/audio/a.wav",
                    "subject": "nails",
                    "verb": "tapping",
                    "object": "glass bottle",
                },
                {
                    "wav": "test/audio/b.wav",
                    "subject": "hand",
                    "verb": "rubbing",
                    "object": None,
                },
            ]
        }
        manual_gt = {
            "data": [
                {
                    "item_id": "a",
                    "wav": "test/audio/a.wav",
                    "gt": {"subject": "fingernails", "verb": "patting", "object": "glass bottle"},
                    "annotation": {"annotator_id": "a", "status": "complete"},
                },
                {
                    "item_id": "b",
                    "wav": "test/audio/b.wav",
                    "gt": {"subject": "hand", "verb": "scratching", "object": None},
                    "annotation": {"annotator_id": "a", "status": "complete"},
                },
            ]
        }
        aliases = "group_id,alias\nfingernails,fingernails\nfingernails,nails\n"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            prediction_path = work / "predictions.json"
            gt_path = work / "manual_gt.json"
            alias_path = work / "noun_alias.csv"
            out_path = work / "quality.json"
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            gt_path.write_text(json.dumps(manual_gt), encoding="utf-8")
            alias_path.write_text(aliases, encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "evaluation/annotation_quality/evaluate_annotations.py"),
                    "--predictions",
                    str(prediction_path),
                    "--ground-truth",
                    str(gt_path),
                    "--noun-aliases",
                    str(alias_path),
                    "--out",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            score = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(score["verb"]["strict"]["accuracy"], 0.0)
            self.assertEqual(score["verb"]["alias_normalized_fine"]["accuracy"], 0.5)
            self.assertEqual(score["verb"]["superclass"]["accuracy"], 1.0)
            self.assertEqual(score["subject"]["strict_accuracy"], 0.5)
            self.assertEqual(score["subject"]["alias_accuracy"], 1.0)
            self.assertEqual(score["object"]["strict_accuracy"], 1.0)
            self.assertIn("manually annotated", score["ground_truth_requirement"])


if __name__ == "__main__":
    unittest.main()
