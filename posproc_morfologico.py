# -*- coding: utf-8 -*-
"""
PÓS-PROCESSAMENTO MORFOLÓGICO + REAVALIAÇÃO POR ÁREA
====================================================
As deteções do modelo saem "roídas": buracos internos e fragmentos por causa
da faixa brilhante da cobertura e do plástico não homogéneo. Isto baixa o
recall de ÁREA (o modelo localiza a estufa mas cobre só parte dela).

Este script tenta recuperar área SEM retreinar, aplicando aos polígonos
detetados, por ordem:
  1. Preenchimento de buracos internos (remove furos dentro de cada estufa)
  2. Fecho morfológico via buffer(+d).buffer(-d): une fragmentos próximos e
     fecha entalhes, sem inflar o contorno externo global
  3. (opcional) dissolver polígonos que se sobrepõem/tocam após o fecho

Depois reavalia por área na sub-zona rigorosa (máscara A), para medir o ganho.
Como a precisão de área estava em 0.97 (muita folga), há margem para "engordar"
as deteções sem começar a apanhar não-estufa.

Uso: editar caminhos + BUFFER_FECHO e correr. Testa vários valores de BUFFER_FECHO.
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# === CONFIGURAÇÃO ===
PRED_SHP    = r"data/resultado_E2_20260807_134608.shp"
VERDADE_SHP = r"data/verdade_terreno_estufas_mascara_A.gpkg"
MASCARA_SHP = r"data/verdade_terreno_area_mascara_A.gpkg"

BUFFER_FECHO = 2.0    # metros. Testa 0 (só buracos), 1, 2, 3, 5. Maior = une mais mas arrisca fundir estufas vizinhas
PREENCHER_BURACOS = True
DISSOLVER = True      # fundir polígonos que passem a tocar-se após o fecho
GRAVAR_SHP = True

# === FUNÇÕES DE LIMPEZA ===
def preencher_buracos(geom):
    """Remove furos internos de Polygon/MultiPolygon (mantém só o anel exterior)."""
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def fecho(geom, d):
    """Fecho morfológico: dilata d e depois erode d. Une fragmentos e fecha
    entalhes estreitos (<2d) sem aumentar o contorno externo global."""
    if d <= 0:
        return geom
    return geom.buffer(d, join_style=2).buffer(-d, join_style=2)

# === CARREGAR ===
pred = gpd.read_file(PRED_SHP)
gt = gpd.read_file(VERDADE_SHP)
mask = gpd.read_file(MASCARA_SHP)
if gt.crs != pred.crs:  gt = gt.to_crs(pred.crs)
if mask.crs != pred.crs: mask = mask.to_crs(pred.crs)
mask_u = mask.union_all()

print(f"Deteções originais: {len(pred)}")

# === APLICAR PÓS-PROCESSAMENTO ===
geoms = list(pred.geometry)
if PREENCHER_BURACOS:
    geoms = [preencher_buracos(g) for g in geoms]
geoms = [fecho(g, BUFFER_FECHO) for g in geoms]
geoms = [g for g in geoms if g is not None and not g.is_empty]

if DISSOLVER:
    dissolvido = unary_union(geoms)
    if isinstance(dissolvido, Polygon):
        geoms = [dissolvido]
    else:
        geoms = list(dissolvido.geoms)
    # re-preencher buracos que possam ter surgido ao dissolver
    if PREENCHER_BURACOS:
        geoms = [preencher_buracos(g) for g in geoms]

pred_pp = gpd.GeoDataFrame(geometry=geoms, crs=pred.crs)
print(f"Deteções após pós-processamento: {len(pred_pp)} "
      f"(fecho={BUFFER_FECHO}m, buracos={PREENCHER_BURACOS}, dissolver={DISSOLVER})")

# === AVALIAÇÃO POR ÁREA (na sub-zona rigorosa) ===
pred_clip = pred_pp.intersection(mask_u);  pred_clip = pred_clip[~pred_clip.is_empty]
gt_clip = gt.intersection(mask_u);          gt_clip = gt_clip[~gt_clip.is_empty]

pred_union = pred_clip.union_all()
gt_union = gt_clip.union_all()
inter = pred_union.intersection(gt_union).area
area_pred = pred_union.area
area_gt = gt_union.area
union = pred_union.union(gt_union).area

prec = inter / area_pred if area_pred else 0
rec = inter / area_gt if area_gt else 0
f1 = 2*prec*rec/max(prec+rec, 1e-9)

print("=" * 60)
print(f"POR ÁREA (após pós-processamento, fecho={BUFFER_FECHO}m)")
print(f"  Área detetada:        {area_pred/10000:7.2f} ha")
print(f"  Área verdade terreno: {area_gt/10000:7.2f} ha")
print(f"  Área interseção:      {inter/10000:7.2f} ha")
print(f"  Precisão (área): {prec:.3f}")
print(f"  Recall   (área): {rec:.3f}   <<< comparar com 0.468 (sem pós-proc.)")
print(f"  F1       (área): {f1:.3f}")
print(f"  IoU global:      {inter/union if union else 0:.3f}")
print("=" * 60)
print("Se o recall de área subiu sem a precisão cair muito, o fecho ajudou.")
print("Testa vários BUFFER_FECHO (1,2,3,5) e escolhe o melhor compromisso.")

if GRAVAR_SHP:
    out = PRED_SHP.replace('.shp', f'_posproc_fecho{int(BUFFER_FECHO)}m.shp')
    pred_pp.to_file(out)
    print(f"\n✓ Deteções pós-processadas: {out}")
