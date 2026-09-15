# -*- coding: utf-8 -*-
"""
FASE 0 — RECORTE DO ORTOFOTO ÀS MÁSCARAS DE TREINO E TESTE
Produz os TIFFs comprimidos (LZW) a subir ao Google Drive.
Uso: editar os caminhos e correr  python recortar_fase0.py
"""

import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import os

RASTER = r"data/ortofoto_zona_vulneravel.tif"
SAIDA  = r"output/Colab_Estufas2026"   # pasta local de preparação

RECORTES = [
    # (ficheiro da máscara, nome do tiff de saída)
    (r"data/mascara_estufas_treino_v2.gpkg", "ortoSat_treino_v2.tif"),
    (r"data/mascara_teste_v2.gpkg", "ortoSat_teste_v2.tif"),
]

os.makedirs(SAIDA, exist_ok=True)

for shp, nome in RECORTES:
    print(f"\n✂️  {nome} <- {os.path.basename(shp)}")
    gdf = gpd.read_file(shp)
    with rasterio.open(RASTER) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        # crop=True corta ao bbox; o interior fora do polígono fica nodata=0
        img, transform = rio_mask(src, gdf.geometry, crop=True, nodata=0)
        img = img[:3]  # apenas RGB — descarta alfa se existir
        meta = src.meta.copy()
        meta.update(count=3, height=img.shape[1], width=img.shape[2],
                    transform=transform, compress='lzw', nodata=0,
                    tiled=True, blockxsize=256, blockysize=256)
    out = os.path.join(SAIDA, nome)
    with rasterio.open(out, 'w', **meta) as dst:
        dst.write(img)
    mb = os.path.getsize(out) / 1e6
    print(f"   ✓ {img.shape[2]}x{img.shape[1]} px | {mb:.0f} MB -> {out}")

print("\n✅ Recortes prontos. Sobe a pasta para o Drive: MyDrive/Estufas2026/")
print("Junta também: estufas_treino.shp, mascara_treino.shp, negativos_hard.shp,")
print("mascara_teste (deteta_mascara_1).shp, verdade_terreno_mascara1.shp")
print("(cada shapefile = os 4-5 ficheiros: .shp .shx .dbf .prj [.cpg])")
