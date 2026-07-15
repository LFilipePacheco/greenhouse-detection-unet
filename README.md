# Greenhouse detection from satellite imagery with deep learning
### U-Net semantic segmentation · Esposende – Vila do Conde Vulnerable Zone, Portugal

> Automatic mapping of greenhouse structures across a Nitrates Directive
> Vulnerable Zone using a convolutional neural network (U-Net) applied to
> Sentinel-2 imagery — revealing intensive horticulture absent from official
> land-parcel registers.

---

## Why detect greenhouses

Identifying the greenhouses present in the Esposende – Vila do Conde
Vulnerable Zone (*Zona Vulnerável*, ZV1) matters because intensive
horticulture is one of the main sources of aquifer contamination by
nitrates. At the same time, there were grounds to suspect that most
growers do not register their greenhouse parcels in the official
land-parcel identification system (iSIP, managed by IFAP) — a suspicion the
results confirmed: **1,209 detected greenhouses do not intersect any parcel
registered as "protected crops"**, quantifying for the first time the
extent of under-registration in the zone.

A complete, up-to-date greenhouse layer is therefore both a compliance
instrument and a risk-mapping input: it can be crossed with land use,
farming practices and nitrate concentration data to target monitoring and
enforcement where pressure on groundwater is highest.

Manual mapping at this scale is impractical — the zone spans thousands of
hectares across three municipalities, and greenhouse stock changes from
season to season. The task called for automation.

## The approach

Detection was performed automatically from satellite imagery using an
artificial-intelligence model specialised in image pattern recognition — a
convolutional neural network of the **U-Net** type, widely used in remote
sensing for semantic segmentation. The process ran in three stages:

### 1. Learning (model training) — `train_unet.py`

A set of greenhouses previously identified and validated in a GIS
environment served as ground truth to "teach" the model to recognise these
structures in Sentinel-2 composites. To increase robustness and
generalisation:

- the training set was enriched with **image augmentation** (rotations,
  flips, brightness adjustments);
- **negative examples** — areas without greenhouses — were deliberately
  included so the model would not confuse greenhouses with other bright,
  regular landscape elements. This proved essential given the large number
  of **swimming pools**, particularly around Esposende, whose spectral
  signature is a classic confounder;
- training used loss functions designed for **strongly imbalanced
  segmentation** (greenhouses occupy a tiny fraction of any scene): a
  weighted combination of Dice loss and Focal Tversky loss (0.7/0.3),
  monitored with the Dice coefficient.

The architecture is a classical U-Net: a contracting encoder (64 → 1024
filters), a symmetric decoder, and skip connections that let the network
combine coarse context with fine spatial detail — the property that makes
U-Net effective at delineating small, sharp-edged structures.

### 2. Application (inference) — `detect_greenhouses.py`

The trained model was applied to the entire Vulnerable Zone, processing the
image systematically in overlapping 256-pixel blocks (sliding window with
64-pixel overlap, predictions averaged in overlap zones to avoid edge
artefacts). Band values are normalised per scene using robust percentiles
(2–98). For every point in the territory the model outputs a probability of
belonging to a greenhouse; points above a confidence threshold are
converted into georeferenced polygons.

### 3. Validation and filtering

The resulting polygons were subjected to **area criteria (40 m² –
3,000 m²)** to exclude noise and structures clearly incompatible with
greenhouses, then **visually inspected in a GIS environment** to confirm
the results. Morphological operations (opening/closing) clean the binary
mask before vectorisation, and polygon geometry is simplified for practical
GIS use.

## The result

A georeferenced vector layer (shapefile) with the location and area of every
detected greenhouse — ready to be crossed with the other themes relevant to
Vulnerable Zone monitoring: land use, farming practices, parcel registration
status and nitrate contamination risk.

The headline finding: **1,209 detected greenhouses without a matching
"protected crops" parcel registration**, evidencing systematic
under-registration in iSIP-IFAP and giving the monitoring authority, for
the first time, an independent and repeatable census of protected
horticulture in the zone.

## Repository contents

| File | Purpose |
|---|---|
| `train_unet.py` | Model training: data preparation from GIS ground truth, augmentation, U-Net construction, specialised losses, checkpointing |
| `detect_greenhouses.py` | Inference over large rasters: sliding window, thresholding, morphology, vectorisation, area filtering, diagnostic plots |
| `requirements.txt` | Python dependencies |

Paths in both scripts are placeholders (`path/to/...`) — point them at your
own imagery, ground truth and output folder.

## Stack

Python · TensorFlow/Keras · rasterio · GeoPandas · OpenCV · Shapely ·
scikit-learn · Sentinel-2 L2A imagery (Copernicus)

## About the data and the model

The satellite imagery is open data, but the ground-truth
polygons, the trained model weights and the detection results are
institutional property of CCDR-Norte, I.P. and are not published here. The
code is shared as a working reference implementation of the full
train-and-detect pipeline.

---

**Luís Filipe Pacheco** — Senior Agricultural Engineer & Data Scientist,
CCDR-Norte, I.P. · [GitHub profile](https://github.com/LFilipePacheco) ·
[LinkedIn](https://www.linkedin.com/in/lu%C3%ADs-filipe-pacheco-471495b/) ·
[ORCID](https://orcid.org/0009-0001-7676-6542)
