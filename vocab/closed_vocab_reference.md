# Closed vocabularies (reference)

This file summarizes the **taxonomy** used by DeepASMR-NSpeech and SVO-AQA.
Full CSV tables, alias maps, LLM prompts, and evaluation scripts are in the
**GitHub code repository** (not duplicated here).

---

## Verb taxonomy · 37 verbs → 18 superclasses

### Single-verb superclasses (1 verb each)

- **bubbling** → `bubbling` — Continuous gentle gas bubbles in liquid or wet foam—not solid scraping or crushing.
- **blowing** → `blowing` — Directed airflow or breath on a surface—no sustained solid–solid slide.
- **sizzling** → `sizzling` — Heated oily food micro-pops from cooking—not dry flame crackle alone.
- **crackling** → `crackling` — Flame, combustion, or dry thermal crackle—not candy pop or package snap.
- **popping** → `popping` — Sharp discrete micro-pops (candy, bubble-wrap, gel-pod)—not continuous crackle or crush.
- **rotating** → `rotating` — Object spins in place with rotation/mechanical whirr—not flip of pages/cards.
- **flipping** → `flipping` — Repeated flip/turn cyclic contacts (cards, pages)—not continuous rotation hum.
- **ticking** → `ticking` — Regular mechanical tick from clock/timer escapement—not keyboard typing.
- **tinkling** → `tinkling` — Metal/glass/crystal parts of the prop itself lightly colliding or ringing—object-intrinsic crisp shimmer (like crackling is flame-intrinsic); not tool-on-surface scrape, pour/spill, or shaken-container rattle.

### Multi-verb superclasses (within-class disambiguation applies)

#### `friction_contact`
- **Verbs:** `rubbing`, `scratching`, `brushing`, `rolling`
- **Class cue:** Sustained solid–solid slide, bristle stroke, scratch, or roll—not peel/cut, crush, stir-in-cup, wet smear, or modeling-pack peel/scrape.
- **Disambiguation** (rubbing vs scratching vs brushing vs rolling):
  - rubbing: Two flat surfaces/pads slide with sustained friction (hands, paper, sponge, fabric).
  - scratching: Sharp tip/nail high-frequency friction on a flat surface (earpick tip, nail on mesh).
  - brushing: Many bristles/fibers stroke the surface (spoolie, hair brush, makeup brush).
  - rolling: Sphere, marble, bead, or roller slides or rotates on a surface while contacting it.

#### `impulsive_contact`
- **Verbs:** `tapping`, `typing`, `dropping`
- **Class cue:** Short discrete impacts: tap, key press, drop—NOT sustained slide or crush.
- **Disambiguation** (tapping vs typing vs dropping):
  - tapping: Single or few light discrete impacts.
  - typing: Repeated key/button impacts in sequence.
  - dropping: Object falls and hits a surface.

#### `cutting_penetration`
- **Verbs:** `scraping`, `cutting`, `tearing`, `piercing`
- **Class cue:** Edge/tool removes material; blade cuts, tears, pierces, or scrapes layer off—including modeling pack / face mask peel or scrape from ear or mic—NOT whole-object crush or stir-in-container.
- **Disambiguation** (scraping vs tearing vs cutting vs piercing):
  - scraping: Hard edge/tool shaves or removes material from surface (dried modeling pack + metal tool).
  - tearing: Whole layer/sheet ripped open by pulling force.
  - cutting: Blade slices through material.
  - piercing: Sharp pointed tip penetrates through material—distinct penetration sound (needle through fabric, tip through soft substance), not surface scrape or blade slice.

#### `compression_deformation`
- **Verbs:** `pressing`, `squeezing`, `crushing`, `smashing`, `stretching`
- **Class cue:** Pressure deforms, stretches, or breaks a solid—crinkle, crush, smash—NOT blade peel, wet wipe, or stir inside cup.
- **Disambiguation** (pressing vs squeezing vs crushing vs smashing vs stretching):
  - pressing: Pressure without clear break (paper crinkling, light compress).
  - squeezing: Enclosing soft deformable object.
  - crushing: Hard material or crust collapses under strong force (floral foam, dried flowers, popcorn).
  - smashing: Wet or sticky object being crushed (slime, jelly, floam, orbeez, water beads).
  - stretching: Viscous slime/gel/cotton pulled apart under tension without full layer tear.

#### `vocal_mouth`
- **Verbs:** `whispering`, `crunching`, `licking`
- **Class cue:** Whisper, chew/crunch, or lick—oral cavity only; no external tool on mic.
- **Disambiguation** (whispering vs crunching vs licking):
  - whispering: Breathy low inaudible speech.
  - crunching: Mouth/teeth on food or brittle micro-breakage in oral/eating context.
  - licking: Tongue/lip wet contact.

#### `fluid_transfer`
- **Verbs:** `pouring`, `spraying`
- **Class cue:** Liquid or particle stream between containers or from nozzle—NOT smear on fixed surface.
- **Disambiguation** (pouring vs spraying):
  - pouring: Continuous liquid or granular stream transferred between containers or onto a surface.
  - spraying: Atomized liquid from spray bottle, nozzle, or pressurized source—not a thick pour.

#### `liquid_spread_wipe`
- **Verbs:** `spreading`, `wiping`, `sticking`
- **Class cue:** Wet/sticky medium smeared, wiped, or adhered on a surface—NOT dry modeling-pack peel/scrape, NOT in-cup stirring.
- **Disambiguation** (spreading vs wiping vs sticking):
  - spreading: Paste, cream, gel, or coating smeared/distributed across a surface (sticky wet drag).
  - wiping: Broad pad, cloth, or hand removes or redistributes wet medium on a surface.
  - sticking: Adhesive tape/sticker/paint tack attaches or detaches with sticky pull-release clicks.

#### `container_agitation`
- **Verbs:** `stirring`, `shaking`
- **Class cue:** Stir or shake contents inside a container—NOT external slide on mic fur alone.
- **Disambiguation** (stirring vs shaking):
  - stirring: Circular or repeated agitation of liquid/granules inside a container (spoon, stick).
  - shaking: Oscillatory shake causing contents to rattle, slosh, or impact container walls.

#### `writing_painting`
- **Verbs:** `writing`, `painting`
- **Class cue:** Pen/stylus line-making or brush moving paint on a surface—not ear cleaning friction.
- **Disambiguation** (writing vs painting):
  - writing: Pen, pencil, marker, or stylus line-making friction on paper or hard surface.
  - painting: Brush, palette knife, or tool applies/moves paint across a surface.

---

## Material taxonomy · 12 superclasses

Used in SVO-AQA material multiple-choice questions (`object_material_mc`,
`subject_material_mc`). Clip-level material labels are **not** shipped in
train/test JSON; MC options carry the material gold labels.

Note: the release name is **`clay`** (modeling pack / wet sticky sound),
not `ceramic`.

- **plastic** — stiff polymer body; crisp hollow resonance
- **wood** — woody hardstock; dry grain friction
- **clay** — modeling pack; wet sticky sound
- **foam_lather** — wet liquid lather or shaving foam only; soft squish—not solid bar soap
- **foam_solid** — expanded polymer foam; bead squeak porous sponge
- **glass** — bright brittle crystal body or ice or ceramic; sharp ring transient
- **paper** — light fibrous sheet; crinkle and tear rustle
- **metal** — metallic solid or foil; bright ring transient
- **rubber** — elastic silicone or latex; damped squeak stick-slip
- **wax** — wax or solid bar soap; warm low-mid scrape tap or carve
- **leather** — animal hide skin; flex creak surface rub
- **textile_fibrous** — woven pile fabric; soft cotton wool rustle

---

## GitHub (full artifacts)

For reproducibility, obtain from the project repository:

- `vocab/asmr_verbs_en.csv` — 37 verb short hints
- `vocab/asmr_verb_classes_en.csv` — verb → superclass
- `vocab/asmr_material_classes_compact_en.csv` — 12 material classes
- `vocab/stage2_class_disambiguation.txt` — fine-grained disambiguation rules
- `vocab/verb_alias.csv`, `vocab/material_alias.csv` — alias maps
- Annotation prompts under `annotation/`
