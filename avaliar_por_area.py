# -*- coding: utf-8 -*-
"""
AVALIAÇÃO POR ÁREA (além da por objeto)
=======================================
A avaliação objeto-a-objeto penaliza injustamente quando:
 - uma estufa longa é detetada como vários polígonos (fragmentação),
 - os contornos não coincidem exatamente (IoU < 0.3 apesar de acerto claro).

Este script calcula a métrica por ÁREA, que é imune a esses problemas e mede
o que realmente interessa num mapeamento: que fração da superfície de estufa
foi corretamente identificada, e que fração da área detetada é mesmo estufa.

Métricas por área:
  Precisão_área = área(deteções ∩ verdade) / área(deteções)
  Recall_área   = área(deteções ∩ verdade) / área(verdade)
  IoU_global    = área(interseção) / área(união)

Recorta tudo à máscara antes de medir (mesma correção da avaliação por objeto).
Uso: editar caminhos e correr  python avaliar_por_area.py
"""

import geopandas as gpd

# === CONFIGURAÇÃO ===
PRED_SHP    = r"data/resultado_E2_20260807_134608.shp"
VERDADE_SHP = r"data/verdade_terreno_v2.gpkg"
MASCARA_SHP = r"data/mascara_teste_v2.gpkg"  # a de 488 ha validada
NOME_ENSAIO = "E2"

# === CARREGAR E ALINHAR CRS ===
pred = gpd.read_file(PRED_SHP)
gt = gpd.read_file(VERDADE_SHP)
mask = gpd.read_file(MASCARA_SHP)
for nome, g in [("verdade", gt), ("máscara", mask)]:
    if g.crs != pred.crs:
        print(f"  a reprojetar {nome} para {pred.crs}")
if gt.crs != pred.crs:
    gt = gt.to_crs(pred.crs)
if mask.crs != pred.crs:
    mask = mask.to_crs(pred.crs)

mask_u = mask.union_all()

# === RECORTAR À MÁSCARA (interseção geométrica, não por centroide) ===
# Para área, recortamos as próprias geometrias à máscara, para não contar
# área que esteja fora da zona de teste.
pred_clip = pred.intersection(mask_u)
pred_clip = pred_clip[~pred_clip.is_empty]
gt_clip = gt.intersection(mask_u)
gt_clip = gt_clip[~gt_clip.is_empty]

area_pred = pred_clip.area.sum()
area_gt = gt_clip.area.sum()

# uniões dissolvidas (para não contar sobreposições internas em dobro)
pred_union = pred_clip.union_all()
gt_union = gt_clip.union_all()

inter = pred_union.intersection(gt_union).area
union = pred_union.union(gt_union).area

prec_area = inter / area_pred if area_pred else 0
rec_area = inter / area_gt if area_gt else 0
f1_area = 2 * prec_area * rec_area / max(prec_area + rec_area, 1e-9)
iou_global = inter / union if union else 0

print("=" * 60)
print(f"AVALIAÇÃO POR ÁREA — {NOME_ENSAIO}")
print("=" * 60)
print(f"  Área detetada:        {area_pred/10000:8.2f} ha")
print(f"  Área verdade terreno: {area_gt/10000:8.2f} ha")
print(f"  Área interseção:      {inter/10000:8.2f} ha")
print(f"  ----")
print(f"  Precisão (área): {prec_area:.3f}  (fração da área detetada que é estufa)")
print(f"  Recall   (área): {rec_area:.3f}  (fração da área de estufa detetada)")
print(f"  F1       (área): {f1_area:.3f}")
print(f"  IoU global:      {iou_global:.3f}")
print("=" * 60)
print("\nInterpreta:")
print("  Precisão alta => quase toda a área detetada é mesmo estufa (poucos FP reais)")
print("  Recall alto   => o modelo cobre a maior parte da superfície de estufa")
