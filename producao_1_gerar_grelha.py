# -*- coding: utf-8 -*-
"""
FASE PRODUÇÃO (1/2) — GERAR GRELHA DE TILES SOBRE A ZONA VULNERÁVEL
===================================================================
Cria uma malha regular de quadrados que cobre TODA a ZV, sem buracos nem
sobreposições, recortada ao limite da Zona. Cada tile terá um id único.

Escolha do tamanho: 2000 m dá tiles de ~400 ha. À resolução de 30 cm, um tile
de 2 km = ~6667x6667 px, viável em CPU (~15-25 min cada). Baixa para 1500 m se
a memória apertar; sobe para 2500-3000 m para menos tiles (mais lentos cada).

Uso: editar CONFIGURAÇÃO e correr  python producao_1_gerar_grelha.py
Saída: tiles_zv.gpkg  (um polígono por tile, campo 'tile_id')
"""

import geopandas as gpd
from shapely.geometry import box
import numpy as np

# === CONFIGURAÇÃO ===
LIMITE_ZV = r"data/limite_ZV.gpkg"  # o limite da ZV
SAIDA     = r"data/tiles_zv.gpkg"
TILE_M    = 2000     # lado do tile em metros
OVERLAP_M = 64       # sobreposição entre tiles (m) para não perder estufas na fronteira

# === GERAR GRELHA ===
zv = gpd.read_file(LIMITE_ZV)
zv_union = zv.union_all()
crs = zv.crs
print(f"Limite ZV: {zv_union.area/10000:.0f} ha | CRS {crs}")

minx, miny, maxx, maxy = zv_union.bounds
passo = TILE_M - OVERLAP_M   # os tiles avançam com sobreposição

tiles = []
tid = 0
y = miny
while y < maxy:
    x = minx
    while x < maxx:
        # tile com pequena margem de sobreposição
        t = box(x, y, x + TILE_M, y + TILE_M)
        # só manter tiles que intersectam a ZV (descarta os totalmente fora)
        if t.intersects(zv_union):
            # recortar o tile ao limite da ZV (evita processar fora)
            t_clip = t.intersection(zv_union)
            if not t_clip.is_empty and t_clip.area > 100:  # ignorar fatias minúsculas
                tiles.append({'tile_id': tid, 'geometry': t_clip})
                tid += 1
        x += passo
    y += passo

gdf_tiles = gpd.GeoDataFrame(tiles, crs=crs)
gdf_tiles.to_file(SAIDA, driver='GPKG')

print(f"✓ {len(gdf_tiles)} tiles gerados (lado {TILE_M} m, sobreposição {OVERLAP_M} m)")
print(f"✓ Gravado: {SAIDA}")
print(f"\nÁrea total dos tiles: {gdf_tiles.union_all().area/10000:.0f} ha "
      f"(deve ≈ área da ZV: {zv_union.area/10000:.0f} ha)")
print(f"\nEstimativa de tempo de inferência: ~{len(gdf_tiles)*20} min "
      f"(~20 min/tile em CPU) = ~{len(gdf_tiles)*20/60:.1f} h")
print("Abre tiles_zv.gpkg no QGIS para confirmar que a grelha cobre toda a ZV.")
