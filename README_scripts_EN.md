## Scripts - greenhouse detection pipeline

Reference implementation of the training and detection pipeline. The paths at the top of each script are **examples** - adjust them to your environment (`data/`, `models/`, and `output/` directories).

### Usage order

**1. Data preparation**
- `recortar_fase0.py` - crops the orthophoto to the training and test masks (LZW compression), producing the working GeoTIFF files.

**2. Training** (Google Colab, T4 GPU)
- `treino_estufas_colab_comentado.py` - trains the E0 model (U-Net). Commented reference version.
- `treino_estufas_E2_colab.py` - trains the E2 model (U-Net with a ResNet34 encoder pretrained on ImageNet).

**3. Single-area detection** (local, CPU)
- `deteta_estufas_DL_completo_v2.py` - applies the model to a single mask/area using a sliding window. Includes the preprocessing corrections (`/255` normalization and polygon clipping).

**4. Production at Vulnerable Zone scale** (local, CPU, resumable)
- `producao_1_gerar_grelha.py` - generates the tile grid covering the zone.
- `producao_2_inferir_tiles.py` - performs batch inference on all tiles and consolidates the results. Area filter: minimum 40 m², **no maximum area limit**.

**5. Evaluation**
- `avaliar_detecao.py` - object-level evaluation (IoU ≥ 0.3), clipping the ground truth to the mask; generates diagnostic layers (TP/FP and FN).
- `avaliar_por_area.py` - area-based evaluation (area precision, recall, and F1; global IoU), robust to polygon fragmentation.
- `posproc_morfologico.py` - morphological post-processing test (hole filling and closing). Documented as a dead end: it has no effect on area recall after removing the maximum area limit.

### Final model parameters (E0)

`/255` normalization (training and inference) · 256×256 patch · 192 stride · 0.4 threshold · geometric filter disabled · **no maximum area limit** (minimum 40 m²) · loss: 0.7·Dice + 0.3·FocalTversky · Adam, learning rate `5e-4`.

Detection unit: **contiguous block of plastic-covered area**.

### Dependencies

Python · TensorFlow/Keras · rasterio · GeoPandas · OpenCV · Shapely · scikit-learn · E2: segmentation-models

### Data note

The data (orthophoto, model weights, and ground truth) are not included. The orthoSat2023 orthophoto is an open service provided by Portugal's Directorate-General for Territory (DGT). The remaining data are institutional property of CCDR-Norte.
