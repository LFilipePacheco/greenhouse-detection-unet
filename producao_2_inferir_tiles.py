# -*- coding: utf-8 -*-
"""
FASE PRODUÇÃO (2/2) — INFERÊNCIA EM LOTE SOBRE TODOS OS TILES + CONSOLIDAÇÃO
============================================================================
Corre o modelo E0 sobre cada tile da grelha, junta tudo num único shapefile e
resolve as estufas cortadas nas fronteiras (dissolve das que se tocam).

Usa a receita validada do E0: normalização /255, threshold 0.4, filtro
geométrico DESLIGADO, filtro de área 40-3000 m².

Robusto a interrupções: grava o resultado de cada tile num ficheiro próprio e
salta os já feitos. Se a máquina desligar, é só correr de novo — retoma.

Uso: editar CONFIGURAÇÃO e correr  python producao_2_inferir_tiles.py
Saída: estufas_ZV_completo.gpkg  (todas as estufas da Zona Vulnerável)
"""

import os, glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import tensorflow as tf
from tensorflow import keras
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === CONFIGURAÇÃO ===
RASTER    = r"data/ortofoto_zona_vulneravel.tif"
MODEL     = r"models/modelo_E0_20260805_110312.h5"
TILES     = r"data/tiles_zv.gpkg"
OUT_DIR   = r"output/producao_tiles"  # resultados por tile
OUT_FINAL = r"data/estufas_ZV_completo.gpkg"

# Receita validada do E0
PATCH_SIZE = 256
THRESHOLD = 0.4
MIN_AREA_M2 = 40
MAX_AREA_M2 = None   # SEM TETO: o teto 3000 cortava blocos de tuneis contiguos (estufas grandes reais)
DISSOLVE_FINAL = True   # fundir estufas cortadas entre tiles

os.makedirs(OUT_DIR, exist_ok=True)

# === CARREGAR MODELO ===
print("A carregar modelo...")
model = keras.models.load_model(MODEL, compile=False)
print(f"✓ {model.count_params():,} parâmetros")

# === FUNÇÃO DE DETEÇÃO NUM TILE ===
def detetar_tile(tile_geom, tile_id):
    with rasterio.open(RASTER) as src:
        minx, miny, maxx, maxy = tile_geom.bounds
        row_min, col_min = src.index(minx, maxy)
        row_max, col_max = src.index(maxx, miny)
        col_min, row_min = max(0, col_min), max(0, row_min)
        col_max, row_max = min(src.width, col_max), min(src.height, row_max)
        if col_max <= col_min or row_max <= row_min:
            return gpd.GeoDataFrame(geometry=[], crs=src.crs)
        win = Window(col_min, row_min, col_max-col_min, row_max-row_min)
        img = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
        transform = src.window_transform(win)
        crs = src.crs

    img_norm = img.astype('float32') / 255.0     # /255 — coerente com o treino
    h, w = img_norm.shape[:2]
    if h < 16 or w < 16:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    ps, stride = PATCH_SIZE, PATCH_SIZE - 64
    pred = np.zeros((h, w), 'float32'); cnt = np.zeros((h, w), 'float32')

    lote, coords = [], []
    def flush():
        if not lote: return
        preds = model.predict(np.array(lote), verbose=0)
        for pz, (yy, xx) in zip(preds, coords):
            ye, xe = min(yy+ps, h), min(xx+ps, w)
            pred[yy:ye, xx:xe] += pz[:ye-yy, :xe-xx, 0]
            cnt[yy:ye, xx:xe] += 1
        lote.clear(); coords.clear()

    for i in range(0, max(1, h-ps+1), stride):
        for j in range(0, max(1, w-ps+1), stride):
            patch = img_norm[i:min(i+ps,h), j:min(j+ps,w)]
            if patch.shape[:2] != (ps, ps):
                pad = np.zeros((ps, ps, 3), 'float32')
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            lote.append(patch); coords.append((i, j))
            if len(lote) == 16: flush()
    flush()
    cnt[cnt == 0] = 1; pred /= cnt

    # recortar ao polígono do tile (já recortado à ZV na grelha)
    poly_mask = rasterize([(tile_geom, 1)], out_shape=(h, w), transform=transform,
                          fill=0, dtype='uint8')
    pred *= poly_mask

    binm = (pred > THRESHOLD).astype('uint8')
    k = np.ones((3, 3), 'uint8')
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for c in contours:
        if len(c) < 3: continue
        pts = [transform * (pt[0][0], pt[0][1]) for pt in c]
        try:
            poly = Polygon(pts).simplify(1.0, preserve_topology=True)
        except Exception:
            continue
        a = poly.area
        if a >= MIN_AREA_M2 and (MAX_AREA_M2 is None or a <= MAX_AREA_M2):
            polys.append(poly)
    return gpd.GeoDataFrame({'geometry': polys}, crs=crs)

# === LOOP SOBRE TILES (retomável) ===
tiles = gpd.read_file(TILES)
print(f"{len(tiles)} tiles a processar")
for _, trow in tqdm(tiles.iterrows(), total=len(tiles)):
    tid = int(trow['tile_id'])
    out_tile = os.path.join(OUT_DIR, f'tile_{tid:04d}.gpkg')
    if os.path.exists(out_tile):
        continue   # já feito — salta (retoma após interrupção)
    gdf = detetar_tile(trow.geometry, tid)
    # grava sempre (mesmo vazio) para marcar como feito
    if len(gdf) == 0:
        gdf = gpd.GeoDataFrame({'geometry': []}, crs=tiles.crs)
    gdf.to_file(out_tile, driver='GPKG')

# === CONSOLIDAÇÃO ===
print("\nA consolidar todos os tiles...")
partes = []
for f in sorted(glob.glob(os.path.join(OUT_DIR, 'tile_*.gpkg'))):
    g = gpd.read_file(f)
    if len(g) > 0:
        partes.append(g)
if not partes:
    print("⚠ Nenhuma deteção."); raise SystemExit
todos = gpd.GeoDataFrame(gpd.pd.concat(partes, ignore_index=True), crs=partes[0].crs)
print(f"  {len(todos)} deteções brutas (com duplicados nas fronteiras)")

if DISSOLVE_FINAL:
    # fundir estufas que se tocam/sobrepõem nas fronteiras dos tiles
    fundido = unary_union(list(todos.geometry))
    geoms = [fundido] if fundido.geom_type == 'Polygon' else list(fundido.geoms)
    # re-filtrar por área após fusão
    geoms = [g for g in geoms if g.area >= MIN_AREA_M2]  # sem teto (blocos contiguos = 1 estufa)
    todos = gpd.GeoDataFrame({'geometry': geoms}, crs=todos.crs)
    print(f"  {len(todos)} estufas após fusão de fronteiras")

todos['id'] = range(len(todos))
todos['area_m2'] = todos.geometry.area
todos.to_file(OUT_FINAL, driver='GPKG')

print(f"\n✓ RESULTADO FINAL: {OUT_FINAL}")
print(f"  Total de estufas: {len(todos)}")
print(f"  Área total detetada: {todos['area_m2'].sum()/10000:.1f} ha")
print(f"  (lembrar: recall de área ~0.47 => área REAL estimada ~{todos['area_m2'].sum()/10000/0.47:.0f} ha)")
