"""
CARREGAR E USAR MODELO - VERSÃO CORRIGIDA
Resolve o erro de carregamento do modelo
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import tensorflow as tf
from tensorflow import keras
import cv2
from shapely.geometry import Polygon, box
import os
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === CONFIGURAÇÃO ===
RASTER_PATH = r"path/to/sentinel2_composite.tiff"  # multiband Sentinel-2 L2A composite
MODEL_PATH = r"path/to/trained_model.h5"           # output of train_unet.py
OUTPUT_DIR = r"path/to/output"

# Área de teste
USE_MASK = True
MASK_PATH = r"path/to/processing_area.shp"         # optional area of interest

# Parâmetros
PATCH_SIZE = 256
BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.3  # 0.3 (baixo)para detectar mais
MIN_AREA_M2 = 40
MAX_AREA_M2 = 3000

print("=" * 70)
print("APLICAÇÃO DO MODELO - VERSÃO CORRIGIDA")
print("=" * 70)

# === FUNÇÕES ===

def normalize_image(image):
    """Normaliza imagem para [0, 1]"""
    image = image.astype(np.float32)
    for i in range(image.shape[-1]):
        band = image[:, :, i]
        if np.any(band > 0):
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            if p98 > p2:
                image[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    return image

# === CARREGAR MODELO CORRETAMENTE ===

print("\n📦 Carregando modelo...")

# Método 1: Tentar carregar sem compilar (mais seguro)
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Modelo carregado sem compilação")
    model_loaded = True
except Exception as e1:
    print(f"⚠ Erro método 1: {e1}")
    
    # Método 2: Com funções dummy
    try:
        def dummy_loss(y_true, y_pred):
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        def dummy_metric(y_true, y_pred):
            return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y_true), tf.float32))
        
        custom_objects = {
            'combined_loss': dummy_loss,
            'combined_loss_improved': dummy_loss,
            'dice_coefficient': dummy_metric,
            'dice_loss': dummy_loss,
            'focal_loss': dummy_loss,
            'binary_accuracy': dummy_metric
        }
        
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("✓ Modelo carregado com funções dummy")
        model_loaded = True
    except Exception as e2:
        print(f"❌ Erro método 2: {e2}")
        model_loaded = False

if not model_loaded:
    print("\n❌ ERRO: Não foi possível carregar o modelo!")
    print("\nSOLUÇÃO: Execute o script abaixo primeiro:")
    print("-" * 50)
    print("""
import tensorflow as tf
from tensorflow import keras

# Carregar modelo antigo
model = keras.models.load_model(
    r"path/to/old_model.h5",
    compile=False
)

# Salvar sem optimizer
model.save(
    r"path/to/clean_model.h5",
    include_optimizer=False
)

print("Modelo limpo salvo!")
    """)
    print("-" * 50)
    exit(1)

print(f"  Arquitetura: {model.count_params():,} parâmetros")

# === DEFINIR ÁREA ===

print("\n📍 Definindo área de processamento...")

if USE_MASK and os.path.exists(MASK_PATH):
    mask_gdf = gpd.read_file(MASK_PATH)
    
    with rasterio.open(RASTER_PATH) as src:
        if mask_gdf.crs != src.crs:
            mask_gdf = mask_gdf.to_crs(src.crs)
        
        bounds = mask_gdf.total_bounds
        col_min, row_min = src.index(bounds[0], bounds[3])
        col_max, row_max = src.index(bounds[2], bounds[1])
        
        col_min = max(0, col_min)
        row_min = max(0, row_min)
        col_max = min(src.width, col_max)
        row_max = min(src.height, row_max)
        
        TEST_AREA = Window(col_min, row_min, col_max - col_min, row_max - row_min)
        print(f"  Área da máscara: {TEST_AREA.width}x{TEST_AREA.height} pixels")
else:
    TEST_AREA = Window(10000, 10000, 2000, 2000)
    print(f"  Área padrão: {TEST_AREA.width}x{TEST_AREA.height} pixels")

# === PROCESSAR ÁREA ===

print("\n🔍 Detectando estufas...")

with rasterio.open(RASTER_PATH) as src:
    image = src.read(window=TEST_AREA)
    image = np.transpose(image, (1, 2, 0))
    image_norm = normalize_image(image)
    
    transform = src.window_transform(TEST_AREA)
    bounds = src.window_bounds(TEST_AREA)
    crs = src.crs
    pixel_size = src.res[0]
    
    area_ha = (TEST_AREA.width * pixel_size * TEST_AREA.height * pixel_size) / 10000
    print(f"  Área: ~{area_ha:.0f} hectares")

# Aplicar modelo com sliding window
height, width = image_norm.shape[:2]
stride = PATCH_SIZE - 64

prediction = np.zeros((height, width), dtype=np.float32)
counts = np.zeros((height, width), dtype=np.float32)

n_patches_y = max(1, (height - PATCH_SIZE) // stride + 1)
n_patches_x = max(1, (width - PATCH_SIZE) // stride + 1)
total_patches = n_patches_y * n_patches_x

print(f"  Processando {total_patches} patches...")

batch_patches = []
batch_coords = []

with tqdm(total=total_patches) as pbar:
    for i in range(0, max(1, height - PATCH_SIZE + 1), stride):
        for j in range(0, max(1, width - PATCH_SIZE + 1), stride):
            # Garantir que não saia dos limites
            i_end = min(i + PATCH_SIZE, height)
            j_end = min(j + PATCH_SIZE, width)
            
            # Se o patch for menor que PATCH_SIZE, fazer padding
            patch = image_norm[i:i_end, j:j_end]
            
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 4), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            batch_patches.append(patch)
            batch_coords.append((i, j))
            
            # Processar batch
            if len(batch_patches) == BATCH_SIZE or (i >= height - PATCH_SIZE - stride and j >= width - PATCH_SIZE - stride):
                batch_array = np.array(batch_patches)
                
                try:
                    preds = model.predict(batch_array, verbose=0)
                except:
                    # Se falhar, tentar um por vez
                    preds = []
                    for p in batch_array:
                        pred = model.predict(np.expand_dims(p, 0), verbose=0)
                        preds.append(pred[0])
                    preds = np.array(preds)
                
                for pred, (y, x) in zip(preds, batch_coords):
                    y_end = min(y + PATCH_SIZE, height)
                    x_end = min(x + PATCH_SIZE, width)
                    pred_crop = pred[:y_end-y, :x_end-x, 0]
                    
                    prediction[y:y_end, x:x_end] += pred_crop
                    counts[y:y_end, x:x_end] += 1
                
                batch_patches = []
                batch_coords = []
                pbar.update(len(preds))

# Calcular média
counts[counts == 0] = 1
prediction = prediction / counts

print(f"✓ Predição concluída")
print(f"  Probabilidades - Min: {prediction.min():.3f}, Max: {prediction.max():.3f}, Média: {prediction.mean():.3f}")

# === PÓS-PROCESSAMENTO ===

print("\n🔧 Pós-processamento...")

# Aplicar threshold
binary_mask = (prediction > CONFIDENCE_THRESHOLD).astype(np.uint8)

# Operações morfológicas
kernel = np.ones((3, 3), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# Encontrar contornos
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"✓ {len(contours)} objetos encontrados")

# === CRIAR POLÍGONOS ===

print("\n📐 Extraindo polígonos...")

polygons = []
rejected = {'small': 0, 'large': 0}

for contour in contours:
    if len(contour) >= 3:
        coords = []
        for point in contour:
            col, row = point[0]
            x, y = transform * (col, row)
            coords.append((x, y))
        
        if len(coords) >= 3:
            try:
                poly = Polygon(coords)
                poly = poly.simplify(1.0, preserve_topology=True)
                area = poly.area
                
                if area < MIN_AREA_M2:
                    rejected['small'] += 1
                elif area > MAX_AREA_M2:
                    rejected['large'] += 1
                else:
                    polygons.append(poly)
            except:
                continue

print(f"✓ {len(polygons)} estufas válidas")
if rejected['small'] > 0:
    print(f"  {rejected['small']} rejeitadas por serem pequenas (<{MIN_AREA_M2} m²)")
if rejected['large'] > 0:
    print(f"  {rejected['large']} rejeitadas por serem grandes (>{MAX_AREA_M2} m²)")

# === SALVAR RESULTADOS ===

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Sempre salvar visualização
print("\n📊 Salvando visualização...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image[:, :, :3].astype(np.uint8))
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

im = axes[0, 1].imshow(prediction, cmap='hot', vmin=0, vmax=1)
axes[0, 1].set_title(f'Probabilidades (max={prediction.max():.3f})')
axes[0, 1].axis('off')
plt.colorbar(im, ax=axes[0, 1])

axes[0, 2].imshow(binary_mask, cmap='gray')
axes[0, 2].set_title(f'Máscara (threshold={CONFIDENCE_THRESHOLD})')
axes[0, 2].axis('off')

axes[1, 0].hist(prediction.flatten(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold')
axes[1, 0].set_xlabel('Probabilidade')
axes[1, 0].set_ylabel('Frequência')
axes[1, 0].set_title('Distribuição')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

contour_img = np.zeros_like(binary_mask)
cv2.drawContours(contour_img, contours, -1, 255, 1)
axes[1, 1].imshow(contour_img, cmap='gray')
axes[1, 1].set_title(f'{len(contours)} Objetos')
axes[1, 1].axis('off')

info_text = f"""Estatísticas:
Objetos: {len(contours)}
Estufas válidas: {len(polygons)}
Pequenas: {rejected['small']}
Grandes: {rejected['large']}

Threshold: {CONFIDENCE_THRESHOLD}
Área: {MIN_AREA_M2}-{MAX_AREA_M2} m²

Probs min: {prediction.min():.3f}
Probs max: {prediction.max():.3f}
Probs média: {prediction.mean():.3f}
"""
axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='center', family='monospace')
axes[1, 2].axis('off')

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, f'deteccao_{timestamp}.png')
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Visualização salva: {viz_path}")

# Salvar shapefile se houver detecções
if len(polygons) > 0:
    print("\n💾 Salvando shapefile...")
    
    gdf = gpd.GeoDataFrame({
        'id': range(len(polygons)),
        'area_m2': [p.area for p in polygons],
        'geometry': polygons
    }, crs=crs)
    
    shp_path = os.path.join(OUTPUT_DIR, f'estufas_{timestamp}.shp')
    gdf.to_file(shp_path)
    print(f"✓ Shapefile salvo: {shp_path}")
    
    print(f"\n📊 Resumo:")
    print(f"  Total: {len(gdf)} estufas")
    print(f"  Área média: {gdf['area_m2'].mean():.1f} m²")
    print(f"  Área total: {gdf['area_m2'].sum():.1f} m²")
else:
    print("\n⚠ Nenhuma estufa detectada com os parâmetros atuais")
    print(f"  Sugestão: Reduza CONFIDENCE_THRESHOLD para {max(0.05, CONFIDENCE_THRESHOLD - 0.05)}")

print("\n✅ PROCESSAMENTO CONCLUÍDO!")
print(f"Analise a visualização: {viz_path}")