# Greenhouse detection from satellite imagery with deep learning
### U-Net semantic segmentation · Esposende – Vila do Conde Vulnerable Zone, Portugal

> A reproducible train-and-detect pipeline that maps greenhouse structures
> across a Nitrates Directive Vulnerable Zone (~20,000 ha) from high-resolution
> satellite orthoimagery (DGT *ortoSat2023*, RGB, ~30 cm/pixel). This repository
> documents the **2026 campaign**: a ground-up rework of an earlier pilot,
> built around bug diagnosis, strict train/test separation and rigorous
> ground truth — and an honest account of where the approach works and where
> it hits its ceiling.

---

![Detected greenhouses over aerial imagery](docs/detections_overlay.png)
*Detected greenhouse polygons (yellow) over high-resolution orthoimagery — model output after area filtering and GIS validation.*

![Inference diagnostic panel](docs/inference_panel.png)
*Inference diagnostics for one processing block: area of interest, mask, probability map, binary mask, detections and area distribution.*

## Headline result

Evaluated against **rigorously digitised ground truth** (83 greenhouses over a
~50 ha sub-zone), on the primary task of **locating and counting** greenhouses:

| Task | Precision | Recall | F1 |
|---|---|---|---|
| **Object detection** (location / count) | **0.92** | **0.72** | **0.81** |
| Area coverage | 0.97 | 0.47 | — |

The dual conclusion is the scientific core of the project: the model is
**reliable for locating and counting** greenhouses (the primary objective),
but **underestimates total area by roughly half**, because it misses
greenhouses with degraded or heterogeneous covering. This area limitation
proved irreducible through architecture, more data or post-processing — an
input-data (spectral) limit, not a modelling one.

## Why detect greenhouses

Intensive horticulture under plastic is one of the main sources of aquifer
contamination by nitrates in the Esposende – Vila do Conde Vulnerable Zone
(*Zona Vulnerável*, ZV1). A complete, up-to-date greenhouse layer is both a
compliance instrument and a risk-mapping input: crossed with land use,
farming practices and nitrate concentrations, it lets the monitoring
authority target enforcement where pressure on groundwater is highest.

There were also grounds to suspect substantial under-registration of
greenhouse parcels in the official land-parcel system (iSIP, managed by
IFAP). An initial 2025 pilot pointed in that direction, but was affected by a
high false-positive rate that made any count unreliable — which is precisely
what motivated the rigorous 2026 rework documented here. A validated,
zone-wide census is the natural next step once the corrected model is scaled
to all 19 processing masks (see *Future work*).

Manual mapping at this scale is impractical — thousands of hectares across
three municipalities, with greenhouse stock changing season to season. The
task called for automation.

## From 2025 to 2026 — what changed

The 2025 pilot produced too many false positives (asphalt roads, rooftops),
requiring extensive manual cleaning. Rather than patch it, the 2026 campaign
diagnosed the root causes and rebuilt the pipeline. Three defects were found,
in order of impact:

1. **Inconsistent normalisation between training and inference** — the
   dominant cause of false positives. Training normalised by `/255`; inference
   stretched contrast with percentiles (p2–p98), so the model saw pixels on a
   different scale and classified whole fields as greenhouse. **Fix:** `/255`
   throughout the pipeline. This single change moved object precision from
   **0.13 → 0.85** on the test mask.
2. **Row/column swap** — `rasterio ... index(x, y)` returns `(row, col)`; the
   code unpacked `(col, row)`, transposing the read window.
3. **Bounding box instead of polygon** — prediction covered the mask's
   bounding box (e.g. 701 ha) rather than the true polygon (e.g. 474 ha),
   generating detections outside the area of interest. **Fix:** rasterise the
   polygon and null the exterior.

Reading was also fixed to `src.read([1,2,3])` (RGB), discarding the alpha
channel the WMTS service sometimes exports as a fourth band.

## Method (corrected pipeline)

**Golden rule — preprocessing coherence.** `/255` normalisation and fixed RGB
bands in training, validation and inference, recorded in per-model JSON
metadata. Training and inference must never diverge.

**Inviolable train/test separation.** Training and test zones with no spatial
overlap (geometrically verified). Without this, evaluation measures
memorisation, not generalisation.

**Rigorous ground truth.** Greenhouses digitised manually over the orthophoto
with a consistent criterion (long tunnels with a bright strip as a single
polygon; explicit decisions on low plasticulture). Object-level evaluation at
IoU ≥ 0.3.

**Training data preparation.** 256×256 patches: (a) centred on greenhouses,
(b) complementary sliding window, (c) hard negatives. On-the-fly augmentation
via a `keras.utils.Sequence` generator (patches kept as `uint8`; normalisation
and augmentation per batch) — solving the RAM exhaustion that pre-computed
augmentation caused on Colab.

**Execution environment.** Training on Google Colab (T4 GPU); inference and
evaluation locally (CPU, Windows 11, 32 GB RAM, no NVIDIA GPU). The 19 GB
raster was clipped to each mask (LZW compression) to stay within the free
Google Drive limit — the full raster was never uploaded.

## Experiments and results

Two models were trained and compared:

- **E0 (retrained baseline)** — U-Net (~31 M parameters), combined loss
  (0.7·Dice + 0.3·Focal Tversky), Adam `lr=5e-4`, single training zone.
  `val_dice ≈ 0.94`.
- **E2 (pretrained encoder, recall-oriented)** — U-Net with an ImageNet-
  pretrained **ResNet34** encoder, recall-oriented loss, **multi-zone**
  training (~3× the data of E0).

Object-level evaluation (IoU ≥ 0.3):

| Run | Test zone | GT greenhouses | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Baseline 2025 (with post-proc.) | mask 1 | 33 | 0.13 | 0.76 | 0.23 |
| E0 (no filter, thr 0.4) | mask 1 | 33 | 0.85 | 0.88 | 0.87 |
| E0 (no filter) | mask 2 | 231 | 0.83 | 0.72 | 0.77 |
| E2 (multi-zone) | 488 ha zone | 536 | 0.83 | 0.71 | 0.77 |

Key observations:

- Coherent `/255` normalisation drove precision from **0.13 → 0.85**.
- The small mask (33 greenhouses) was optimistic; the larger zones (231 and
  536 greenhouses) give the reliable figure: **F1 ≈ 0.77**.
- **E0 ≈ E2**: a pretrained encoder plus 3× the data did **not** improve F1.

### Object vs. area — the central finding

Systematically, object recall (~0.72) far exceeds area recall (~0.47): the
model **locates** greenhouses but **covers them only partially**, or detects
the smaller ones and misses large greenhouses with heterogeneous covering. On
the rigorous sub-zone: 10.47 ha real vs. 4.90 ha detected.

### The role of ground-truth quality

The single most determinant factor in the metrics was **not** any model
choice — it was ground-truth quality. Reusing irregular polygons inherited
from the 2025 detection inflated the reference and fragmented long
greenhouses, penalising correct detections as false positives. Digitising
rigorous ground truth from scratch raised object precision **0.83 → 0.92** and
area precision **0.88 → 0.97**, confirming that most apparent "false
positives" were in fact correct detections. Crucially, area recall stayed low
(~0.47) under the rigorous reference too — **ruling out** the hypothesis that
undercoverage was an artefact of the reference.

## What was tested and rejected

Documenting the dead ends is part of the result — it maps the ceiling of the
approach:

- **Confidence threshold** — probabilities are bimodal; tuning the threshold
  had little effect.
- **Geometric anti-road filter** — rejected 16 real greenhouses (narrow
  tunnels are geometrically identical to road segments) while the dominant
  false positives (rooftops) were *more* compact than greenhouses. Turned off;
  the false-positive problem is spectral, not geometric.
- **Deeper / pretrained architecture (E2)** — no F1 gain over E0.
- **3× more training data (multi-zone)** — no F1 gain.
- **Morphological post-processing** (hole filling + closing, d ∈ 0–5 m) — no
  effect on area recall (0.468 throughout), proving the missing area is in
  **whole undetected greenhouses**, not in gaps within detected ones.

## Key lessons

- The biggest gains came from **data hygiene**, not sophisticated
  architecture: coherent normalisation, strict train/test separation, rigorous
  ground truth.
- **Ground-truth quality dominated** every modelling choice. References
  inherited from earlier detections bias evaluation and understate precision.
- Knowing **when to stop** matters: once threshold, architecture, data volume
  and post-processing all failed to move area recall, the limitation was
  correctly attributed to the input signal (degraded-plastic spectral
  signature), not to the model.

## Limitations

- The reliable metrics rest on 231–536 greenhouses (larger zones) and 83 in
  the rigorous sub-zone; the rigorous sub-zone (~50 ha) is small and should be
  enlarged to consolidate paper-grade figures.
- Inter-annotator agreement on the definition of "greenhouse" was not formally
  measured.
- A single image source and epoch (*ortoSat2023*); temporal and cross-sensor
  generalisation not assessed.
- File-version management was a recurrent source of error (same-named masks
  with different areas). Versioned naming and systematic area/overlap checks
  are recommended.

## Future work

- Enlarge rigorous ground truth (several sub-zones, ≥300 greenhouses) for
  robust, paper-grade metrics.
- Inter-annotator agreement study on the "greenhouse" definition.
- Targeted retraining with many degraded/heterogeneous-cover examples — the
  only untested avenue for the area problem, though of uncertain return given
  the flattening data curve.
- Scale the model to all 19 masks (~20,000 ha) for a **validated, zone-wide
  greenhouse census** — the basis for a defensible under-registration figure.
- Explore higher-resolution or multispectral imagery for the physical limit of
  the plastic signature.

## Repository contents

| File | Purpose |
|---|---|
| `deteta_estufas_DL_completo_v2.py` | Local inference (sliding window) with the three fixes and the (switchable) geometric filter. **Recommended:** threshold 0.4, filter off |
| `treino_estufas_colab_comentado.py` | **Reference** training pipeline (E0): U-Net, RAM-constant augmentation generator, integrated inference and evaluation, annotated |
| `treino_estufas_E2_colab.py` | E2 variant: pretrained ResNet34 encoder, recall-oriented loss, multi-zone |
| `recortar_fase0.py` | Clips the orthophoto to training/test masks (LZW) |
| `avaliar_detecao.py` | Object-level evaluation (IoU ≥ 0.3); TP/FP/FN diagnostic layers |
| `avaliar_por_area.py` | Area-level evaluation, robust to polygon fragmentation |
| `pos_processar_e_avaliar.py` | Decoupled post-processing: shape metrics, calibratable filters, threshold search |
| `posproc_morfologico.py` | Hole filling + morphological closing + dissolve, with area re-evaluation |
| `requirements.txt` | Python dependencies |

Each trained model is paired with a JSON metadata file (normalisation, bands,
threshold, loss, seed, metrics). Paths and parameters live in the
configuration section of each script — point them at your own imagery, ground
truth and output folder.

## Reproduction recipe (final model)

- Imagery: *ortoSat2023* (DGT), RGB, ~30 cm/pixel, EPSG:3763
- Normalisation: `/255` (training **and** inference)
- Patch 256×256; sliding-window stride 192 (patch − 64)
- Inference threshold: **0.4**
- Geometric filter: **off**; area filter: 40–3,000 m²
- E0: U-Net ~31 M params, loss 0.7·Dice + 0.3·FocalTversky(0.3/0.5/1.33),
  Adam 5e-4, batch 8, early stopping on `val_dice`
- E2: U-Net + ResNet34 (ImageNet), Tversky(0.6/0.4), Adam 1e-4
- Evaluation: object (IoU ≥ 0.3) and area; ground truth clipped to the mask

**Reference result (rigorous sub-zone, 83 greenhouses, ~50 ha):**
object P=0.92 R=0.72 F1=0.81 · area P=0.97 R=0.47.

## Stack

Python · TensorFlow/Keras · segmentation-models · rasterio · GeoPandas ·
OpenCV · Shapely · scikit-learn · DGT *ortoSat2023* orthoimagery (open WMS,
Direção-Geral do Território)

## About the data and the model

The imagery is the open **ortoSat2023** high-resolution orthoimagery service
of the Direção-Geral do Território (DGT), used with attribution. The
ground-truth polygons, trained model weights and detection results are
institutional property of CCDR-Norte, I.P. and are not published here. The
code is shared as a working reference implementation of the full
train-and-detect pipeline.

---

**Luís Filipe Pacheco** — Senior Engineer & Data Scientist,
CCDR-Norte, I.P. · [GitHub profile](https://github.com/LFilipePacheco) ·
[LinkedIn](https://www.linkedin.com/in/lu%C3%ADs-filipe-pacheco-471495b/) ·
[ORCID](https://orcid.org/0009-0001-7676-6542)
