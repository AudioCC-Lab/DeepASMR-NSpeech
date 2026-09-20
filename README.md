---
title: DeepASMR-NSpeech Demo
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
fullWidth: true
header: mini
pinned: false
short_description: DeepASMR-NSpeech audio, taxonomy, and SVO-AQA demo
---

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

Public entry points:

- Project page: https://bonnie-yzy.github.io/DeepASMR-NSpeech/
- Dataset: https://huggingface.co/datasets/yzyai/DeepASMR-NSpeech
- Code: https://github.com/bonnie-yzy/DeepASMR-NSpeech

Publish the current static demo to the repository's `gh-pages` branch:

```bash
git subtree push --prefix demo origin gh-pages
```

Regenerate the vocabulary tree:

```bash
python3 demo/scripts/generate_vocab_trees.py
```

The static sample media is retained as a paper/demo artifact. The internal
asset-assembly script depended on private model-output and quality-analysis
directories, so it is intentionally not part of the public annotation-code
release.

Code and original documentation are released under MIT. Authorized structured
annotations are released under CC BY-NC 4.0. Copyright in the original source
media remains with the respective original creators or other rights holders.
