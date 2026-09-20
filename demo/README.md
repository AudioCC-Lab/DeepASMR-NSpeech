# DeepASMR-NSpeech demo resources

This directory is a self-contained static project page. It includes the HTML,
sample manifest, example audio/frame assets, annotation/SVO-AQA workflow
figures, and a script that regenerates the vocabulary taxonomy SVG from the
released CSV files.

Preview locally:

```bash
python3 -m http.server 8000 --directory demo
```

Then open `http://localhost:8000`.

Regenerate the vocabulary tree:

```bash
python3 demo/scripts/generate_vocab_trees.py
```

The static sample media is retained as a paper/demo artifact. The internal
asset-assembly script depended on private model-output and quality-analysis
directories, so it is intentionally not part of the public annotation-code
release.
