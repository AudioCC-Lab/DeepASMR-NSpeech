#!/usr/bin/env python3
"""Generate side-by-side verb-class and material-class tree SVGs for the demo page."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERB_CSV = REPO_ROOT / "vocab" / "asmr_verb_classes_en.csv"
MATERIAL_CSV = REPO_ROOT / "vocab" / "asmr_material_classes_en.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"

MATERIAL_RENAME = {"ceramic": "clay"}

MATERIAL_ORDER = [
    "plastic",
    "wood",
    "clay",
    "glass",
    "metal",
    "rubber",
    "leather",
    "wax",
    "paper",
    "textile_fibrous",
    "foam_lather",
    "foam_solid",
]

CLASS_LABEL = {
    "friction_contact": "friction contact",
    "impulsive_contact": "impulsive contact",
    "cutting_penetration": "cutting / penetration",
    "compression_deformation": "compression / deformation",
    "vocal_mouth": "vocal / mouth",
    "fluid_transfer": "fluid transfer",
    "liquid_spread_wipe": "liquid spread / wipe",
    "container_agitation": "container agitation",
    "writing_painting": "writing / painting",
    "textile_fibrous": "textile / fibrous",
    "foam_lather": "foam (lather)",
    "foam_solid": "foam (solid)",
    "clay": "clay (wet sticky)",
}

# Layout constants
SUPERCLASS_GAP = 16
VERB_LINE = 17
CLASS_BOX_H = 22
TRUNK_X_OFFSET = 24


def verb_block_height(n_verbs: int) -> int:
    return max(CLASS_BOX_H, (n_verbs - 1) * VERB_LINE + CLASS_BOX_H)


def verb_ys(cy: int, n_verbs: int) -> list[int]:
    if n_verbs == 1:
        return [cy]
    if n_verbs == 2:
        return [cy - VERB_LINE // 2, cy + VERB_LINE // 2]
    total_span = (n_verbs - 1) * VERB_LINE
    start_y = cy - total_span // 2
    return [start_y + i * VERB_LINE for i in range(n_verbs)]


def draw_class_verbs(
    lines: list[str],
    *,
    class_x: int,
    class_box_w: int,
    verb_stub_x: int,
    verb_x: int,
    box_y: int,
    cy: int,
    label: str,
    verbs: list[str],
) -> None:
    lines.append(
        f'<rect x="{class_x}" y="{box_y}" width="{class_box_w}" height="{CLASS_BOX_H}" rx="7" class="node-class"/>'
    )
    lines.append(f'<text x="{class_x + 10}" y="{box_y + 15}" class="node-class-text">{esc(label)}</text>')

    ys = verb_ys(cy, len(verbs))
    mid_y = int(sum(ys) / len(ys))
    lines.append(
        f'<line x1="{verb_stub_x}" y1="{box_y + CLASS_BOX_H // 2}" x2="{verb_stub_x}" y2="{mid_y}" class="branch-tail"/>'
    )
    lines.append(f'<line x1="{verb_stub_x}" y1="{mid_y}" x2="{verb_x - 6}" y2="{mid_y}" class="branch-tail"/>')

    for verb, vy in zip(verbs, ys):
        lines.append(f'<line x1="{verb_x - 6}" y1="{mid_y}" x2="{verb_x - 6}" y2="{vy}" class="branch-tail"/>')
        lines.append(f'<line x1="{verb_x - 6}" y1="{vy}" x2="{verb_x}" y2="{vy}" class="branch-tail"/>')
        lines.append(f'<text x="{verb_x + 4}" y="{vy + 4}" class="node-leaf">{esc(verb)}</text>')


def load_verbs() -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    with VERB_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["class"].strip()
            verbs = [v.strip() for v in row["verbs"].split(";") if v.strip()]
            rows.append((cls, verbs))
    return rows


def load_materials() -> set[str]:
    names: set[str] = set()
    with MATERIAL_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = MATERIAL_RENAME.get(row["class"].strip(), row["class"].strip())
            names.add(cls)
    return names


def pretty(name: str) -> str:
    return CLASS_LABEL.get(name, name.replace("_", " "))


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def draw_verb_tree(x0: int, y0: int, width: int, verb_rows: list[tuple[str, list[str]]]) -> tuple[str, int, list[tuple[int, int, int]]]:
    """Return SVG lines, bottom y, and trunk segment (x, y1, y2)."""
    lines: list[str] = []
    pad_x = 20
    trunk_x = x0 + pad_x + TRUNK_X_OFFSET
    class_x = trunk_x + 22
    class_box_w = 158
    verb_x = class_x + class_box_w + 18
    verb_stub_x = class_x + class_box_w

    title_y = y0 + 28
    lines.append(
        f'<text x="{x0 + width // 2}" y="{title_y}" text-anchor="middle" class="panel-title">Verb Superclasses → Verbs</text>'
    )
    lines.append(
        f'<text x="{x0 + width // 2}" y="{title_y + 18}" text-anchor="middle" class="panel-sub">37 fine-grained verbs · 18 superclasses</text>'
    )

    y = title_y + 44
    row_centers: list[int] = []

    for cls, verbs in verb_rows:
        label = pretty(cls)
        block_h = verb_block_height(len(verbs))
        cy = y + block_h // 2
        row_centers.append(cy)

        # Trunk → superclass entry (horizontal only to box left edge)
        lines.append(f'<line x1="{trunk_x}" y1="{cy}" x2="{class_x}" y2="{cy}" class="branch"/>')

        box_y = y + (block_h - CLASS_BOX_H) // 2
        draw_class_verbs(
            lines,
            class_x=class_x,
            class_box_w=class_box_w,
            verb_stub_x=verb_stub_x,
            verb_x=verb_x,
            box_y=box_y,
            cy=cy,
            label=label,
            verbs=verbs,
        )

        y += block_h + SUPERCLASS_GAP

    trunk_top = row_centers[0]
    trunk_bottom = row_centers[-1]
    lines.append(f'<line x1="{trunk_x}" y1="{trunk_top}" x2="{trunk_x}" y2="{trunk_bottom}" class="trunk"/>')

    return "\n    ".join(lines), y + 12, (trunk_x, trunk_top, trunk_bottom)


def draw_material_tree(x0: int, y0: int, width: int, materials: set[str]) -> tuple[str, int]:
    lines: list[str] = []
    pad_x = 20
    trunk_x = x0 + pad_x + TRUNK_X_OFFSET
    mat_x = trunk_x + 22
    mat_w = 168
    mat_line = 20

    title_y = y0 + 28
    lines.append(
        f'<text x="{x0 + width // 2}" y="{title_y}" text-anchor="middle" class="panel-title">Material Superclasses</text>'
    )
    lines.append(
        f'<text x="{x0 + width // 2}" y="{title_y + 18}" text-anchor="middle" class="panel-sub">12 closed material classes · SVO-AQA</text>'
    )

    present = [m for m in MATERIAL_ORDER if m in materials]
    y = title_y + 44
    row_centers: list[int] = []

    for mat in present:
        block_h = CLASS_BOX_H
        cy = y + block_h // 2
        row_centers.append(cy)
        box_y = y

        lines.append(f'<line x1="{trunk_x}" y1="{cy}" x2="{mat_x}" y2="{cy}" class="branch"/>')
        lines.append(f'<rect x="{mat_x}" y="{box_y}" width="{mat_w}" height="{CLASS_BOX_H}" rx="7" class="node-leaf-box"/>')
        lines.append(f'<text x="{mat_x + 10}" y="{box_y + 15}" class="node-leaf">{esc(pretty(mat))}</text>')
        y += block_h + SUPERCLASS_GAP

    if row_centers:
        lines.append(
            f'<line x1="{trunk_x}" y1="{row_centers[0]}" x2="{trunk_x}" y2="{row_centers[-1]}" class="trunk"/>'
        )

    return "\n    ".join(lines), y + 12


def build_svg() -> str:
    verb_rows = load_verbs()
    materials = load_materials()
    panel_w = 580
    gap = 36
    total_w = panel_w * 2 + gap + 40
    y0 = 16

    left_body, left_h, _ = draw_verb_tree(20, y0, panel_w, verb_rows)
    right_body, right_h = draw_material_tree(20 + panel_w + gap, y0, panel_w, materials)
    height = max(left_h, right_h, 760) + 8

    mid = 20 + panel_w + gap // 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {height}" font-family="Arial, Helvetica, sans-serif">
  <defs>
    <style>
      .panel {{ fill:#fafbfd; stroke:#e6ebf2; stroke-width:1.5; rx:14; }}
      .panel-title {{ font-size:17px; font-weight:600; fill:#1f2d3d; }}
      .panel-sub {{ font-size:12px; fill:#52667a; }}
      .trunk {{ stroke:#4a6fa5; stroke-width:2; fill:none; }}
      .branch {{ stroke:#4a6fa5; stroke-width:1.5; fill:none; }}
      .branch-tail {{ stroke:#6b8fc0; stroke-width:1.3; fill:none; }}
      .node-class {{ fill:#eef3fb; stroke:#4a6fa5; stroke-width:1.2; }}
      .node-class-text {{ font-size:11.5px; font-weight:600; fill:#224b8d; }}
      .node-group {{ fill:#f3f6fb; stroke:#7a97bf; stroke-width:1.1; }}
      .node-group-text {{ font-size:11px; font-weight:600; fill:#3d5875; }}
      .node-leaf-box {{ fill:#ffffff; stroke:#c9d7ea; stroke-width:1; }}
      .node-leaf {{ font-size:11px; fill:#334155; }}
      .divider {{ stroke:#e6ebf2; stroke-width:1.2; }}
    </style>
  </defs>

  <rect x="10" y="10" width="{panel_w + 20}" height="{height - 20}" class="panel"/>
  <rect x="{20 + panel_w + gap - 10}" y="10" width="{panel_w + 20}" height="{height - 20}" class="panel"/>
  <line x1="{mid}" y1="24" x2="{mid}" y2="{height - 24}" class="divider"/>

  {left_body}
  {right_body}
</svg>
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "vocab_taxonomy_trees.svg"
    out_path.write_text(build_svg(), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
