# -*- coding: utf-8 -*-
"""
AVALIAÇÃO OBJETO-A-OBJETO — deteções vs. verdade de terreno
Uso:  python avaliar_detecao.py
Um TP é uma deteção com IoU >= 0.3 com uma estufa da verdade de terreno
(cada estufa real só conta uma vez).

IMPORTANTE: a verdade de terreno é RECORTADA à máscara antes de avaliar.
Sem isto, estufas reais que existam fora da máscara (onde o modelo nem
processou) seriam contadas como falsos negativos, afundando o recall
artificialmente. Deteções e verdade de terreno têm de cobrir a MESMA área.
"""

import geopandas as gpd
import os

# === CONFIGURAÇÃO ===
PRED_SHP    = r"data/estufas_20260806_150451.shp"
VERDADE_SHP = r"data/verdade_terreno_mascara2.shp"
MASCARA_SHP = r"data/mascara_teste_2.gpkg"  # a MESMA usada na deteção
IOU_MIN     = 0.3
NOME_ENSAIO = "E0_mascara2"

# === CARREGAR ===
gdf_pred = gpd.read_file(PRED_SHP)
gdf_gt = gpd.read_file(VERDADE_SHP)
if gdf_gt.crs != gdf_pred.crs:
    gdf_gt = gdf_gt.to_crs(gdf_pred.crs)

# === RECORTAR A VERDADE DE TERRENO À MÁSCARA (correção crítica) ===
gdf_mask = gpd.read_file(MASCARA_SHP)
if gdf_mask.crs != gdf_pred.crs:
    gdf_mask = gdf_mask.to_crs(gdf_pred.crs)
mask_union = gdf_mask.union_all() if hasattr(gdf_mask, 'union_all') else gdf_mask.unary_union

n_antes = len(gdf_gt)
# manter só as estufas cujo CENTROIDE cai dentro da máscara
# (evita cortar estufas ao meio na fronteira; uma estufa "pertence" à
#  máscara se o seu centro lá estiver)
gdf_gt = gdf_gt[gdf_gt.geometry.centroid.within(mask_union)].reset_index(drop=True)
print(f"Verdade de terreno: {n_antes} total -> {len(gdf_gt)} dentro da máscara")

# opcional: recortar também as deteções à máscara (já deviam estar, mas garante)
n_pred_antes = len(gdf_pred)
gdf_pred = gdf_pred[gdf_pred.geometry.centroid.within(mask_union)].reset_index(drop=True)
if len(gdf_pred) != n_pred_antes:
    print(f"Deteções: {n_pred_antes} -> {len(gdf_pred)} dentro da máscara")

print(f"Deteções: {len(gdf_pred)} | Verdade de terreno (recortada): {len(gdf_gt)}")

gt_usadas = set()
classe = []          # 'TP' ou 'FP' por deteção
iou_valores = []

# usa índice espacial para acelerar
sidx = gdf_gt.sindex

for _, pred in gdf_pred.iterrows():
    melhor_iou, melhor_idx = 0.0, None
    for idx in sidx.intersection(pred.geometry.bounds):
        if idx in gt_usadas:
            continue
        gt_geom = gdf_gt.geometry.iloc[idx]
        inter = pred.geometry.intersection(gt_geom).area
        if inter == 0:
            continue
        union = pred.geometry.union(gt_geom).area
        iou = inter / union
        if iou > melhor_iou:
            melhor_iou, melhor_idx = iou, idx
    if melhor_iou >= IOU_MIN:
        classe.append('TP')
        gt_usadas.add(melhor_idx)
    else:
        classe.append('FP')
    iou_valores.append(round(melhor_iou, 3))

tp = classe.count('TP')
fp = classe.count('FP')
fn = len(gdf_gt) - tp
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-6)

print("\n" + "=" * 60)
print(f"RESULTADO — {NOME_ENSAIO} (IoU >= {IOU_MIN})")
print(f"  TP: {tp}   FP: {fp}   FN: {fn}")
print(f"  Precisão: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")
print("=" * 60)
print("\nLinha para a tabela comparativa:")
print(f"| {NOME_ENSAIO} | {len(gdf_pred)} | {tp} | {fp} | {fn} | "
      f"{prec:.3f} | {rec:.3f} | {f1:.3f} |")

# === SHAPEFILES DE DIAGNÓSTICO ===
gdf_pred['classe'] = classe
gdf_pred['iou'] = iou_valores
out_pred = PRED_SHP.replace('.shp', f'_avaliado_{NOME_ENSAIO}.shp')
gdf_pred.to_file(out_pred)

# estufas reais não detetadas (FN) — para veres o que o modelo perde
gdf_fn = gdf_gt.drop(index=list(gt_usadas))
out_fn = PRED_SHP.replace('.shp', f'_FN_{NOME_ENSAIO}.shp')
if len(gdf_fn) > 0:
    gdf_fn.to_file(out_fn)

print(f"\n✓ Deteções classificadas (campo 'classe' TP/FP): {out_pred}")
if len(gdf_fn) > 0:
    print(f"✓ Estufas não detetadas (FN): {out_fn}")
print("\nNo QGIS: simboliza por 'classe' (TP verde, FP vermelho) e "
      "carrega o shapefile de FN para veres o que falta apanhar.")
