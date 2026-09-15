# -*- coding: utf-8 -*-
"""
TREINO DE DETEÇÃO DE ESTUFAS — GOOGLE COLAB (GPU T4)  [versão comentada]
Zona Vulnerável Esposende–Vila do Conde | ortoSat2023 DGT (RGB, 30 cm)
================================================================================

Cobre as experiências do plano de ação mudando apenas a CÉLULA 2:
  E0 (baseline):  TVERSKY_BETA=0.5, USAR_NEGATIVOS_ESTRADAS=False, ARQUITETURA='unet'
  E1:             TVERSKY_BETA=0.7, USAR_NEGATIVOS_ESTRADAS=True,  ARQUITETURA='unet'
  E2:             como E1 mas ARQUITETURA='efficientnetb0' (ou 'resnet34')

--------------------------------------------------------------------------------
REGRA DE OURO (a lição central do projeto):
A normalização é /255 em TODO o pipeline — treino, validação E inferência. No
ensaio de 2025 a inferência normalizava por percentis e o treino por /255; essa
divergência silenciosa fazia o modelo marcar campos inteiros como estufas. Aqui,
a mesma normalização é usada no treino (célula 5) e na deteção (célula 8), e fica
gravada no JSON de metadados. NUNCA alterar uma sem a outra.
--------------------------------------------------------------------------------

COMO CORRER (dois cenários):

  A) TREINAR UM MODELO NOVO (E0, E1, E2...):
     Runtime > Change runtime type > T4 GPU. Correr as células 1→9 POR ORDEM
     (Runtime > Run all é o mais seguro). A célula 7 é o treino (~30-45 min T4).

  B) SÓ AVALIAR UM MODELO JÁ TREINADO (sem gastar GPU):
     Runtime em CPU. Correr 1, 2, 3, depois RECARREGAR o modelo do Drive
     (ver bloco no fim da célula 7), e SALTAR as células 4, 5, 6, 7. Depois 8, 9.
     A célula 7 (treino) NÃO é necessária — o .h5 já existe no Drive.

AVISOS OPERACIONAIS DO COLAB (fonte da maioria dos erros deste projeto):
  - A sessão reinicia ao mudar de runtime (GPU<->CPU) ou ao desconectar: TODAS as
    variáveis e imports desaparecem. Se saltar células, garantir que as variáveis
    de que as células seguintes dependem (DATA, MODEL_PATH, imports) foram criadas.
  - Correr células fora de ordem gera NameError. Em caso de dúvida, Run all.
  - O ModelCheckpoint grava o melhor modelo DIRETAMENTE no Drive a cada época que
    melhora — se a sessão cair a meio do treino, o melhor modelo não se perde.

RESULTADO DE REFERÊNCIA (E0, máscara 1, 33 estufas GT, IoU>=0.3):
  threshold 0.4, filtro geométrico desligado -> P=0.85 R=0.88 F1=0.87
"""

# %% ============================================================
# CÉLULA 1 — INSTALAÇÃO E DRIVE
# ============================================================
# !pip install -q rasterio geopandas segmentation-models
# import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'   # DESCOMENTAR (E2 usa segmentation_models)
# from google.colab import drive
# drive.mount('/content/drive')

# %% ============================================================
# CÉLULA 2 — CONFIGURAÇÃO DA EXPERIÊNCIA (única célula a editar)
# ============================================================
import os

DATA = '/content/drive/MyDrive/Estufas2026/'

# --- EXPERIÊNCIA E2: encoder pré-treinado + loss orientada ao RECALL ---
# Objetivo: subir o recall (o E0 já tinha boa precisão 0.83, recall 0.72).
# Um encoder ImageNet reconhece mais texturas -> apanha estufas que o U-Net
# de raiz perdia. A loss penaliza mais os falsos negativos (alpha>beta).
EXPERIMENTO = 'E2'
ARQUITETURA = 'resnet34'              # encoder pré-treinado ImageNet
# NOTA: 'efficientnetb0' dá erro 404 ao descarregar pesos (link da biblioteca
# segmentation_models está partido). 'resnet34' usa outra fonte e funciona;
# é uma escolha igualmente boa (ou melhor) para segmentação.
TVERSKY_ALPHA = 0.6                    # penaliza FALSOS NEGATIVOS (>beta => mais recall)
TVERSKY_BETA = 0.4                     # penaliza falsos positivos
TVERSKY_GAMMA = 1.33
USAR_NEGATIVOS_ESTRADAS = False        # FP já controlados; foco no recall

# Ficheiros de TREINO (multi-zona v2 — .gpkg)
TIFF_TREINO   = DATA + 'ortoSat_treino_v2.tif'
SHP_ESTUFAS   = DATA + 'estufas_treino_v2.gpkg'          # estufas de todas as zonas de treino
SHP_MASCARA   = DATA + 'mascara_estufas_treino_v2.gpkg'  # áreas de treino (multipolígono)
SHP_ESTRADAS  = DATA + 'negativos_hard.shp'              # (não usado: USAR_NEGATIVOS=False)

# Ficheiros de TESTE (.gpkg) — zona SEM sobreposição com o treino
TIFF_TESTE    = DATA + 'ortoSat_teste_v2.tif'
SHP_MASC_TESTE= DATA + 'mascara_teste_v2.gpkg'
SHP_VERDADE   = DATA + 'verdade_terreno_v2.gpkg'         # estufas reais na zona de teste

# Hiperparâmetros
PATCH_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 120
LEARNING_RATE = 1e-4                   # menor: encoder pré-treinado precisa de lr baixo
N_AUGMENT = 2
MIN_ESTUFA_PIXELS = 100
FRACAO_NEGATIVOS = 0.33
SEED = 42

# Pós-processamento da inferência
CONFIDENCE_THRESHOLD = 0.4             # valor validado no E0 (filtro geométrico desligado)
MIN_AREA_M2, MAX_AREA_M2 = 40, 3000
MIN_COMPACIDADE, MAX_ALONGAMENTO = 0.0, 9999   # FILTRO GEOMÉTRICO DESLIGADO (prejudica túneis)

# %% ============================================================
# CÉLULA 3 — IMPORTS, SEEDS, LOSSES
# ============================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2, json, warnings
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("GPU:", tf.config.list_physical_devices('GPU') or "NÃO DISPONÍVEL (verificar runtime!)")

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_t = tf.keras.backend.flatten(y_true)
    y_p = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(y_t * y_p)
    return 1 - (2. * inter + smooth) / (tf.keras.backend.sum(y_t) + tf.keras.backend.sum(y_p) + smooth)

def focal_tversky_loss(y_true, y_pred, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, gamma=TVERSKY_GAMMA):
    smooth = 1e-6
    y_t = tf.keras.backend.flatten(y_true)
    y_p = tf.keras.backend.flatten(y_pred)
    tp = tf.keras.backend.sum(y_t * y_p)
    fn = tf.keras.backend.sum(y_t * (1 - y_p))
    fp = tf.keras.backend.sum((1 - y_t) * y_p)
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.keras.backend.pow((1 - tversky), gamma)

def combined_loss(y_true, y_pred):
    return 0.7 * dice_loss(y_true, y_pred) + 0.3 * focal_tversky_loss(y_true, y_pred)

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_t = tf.keras.backend.flatten(y_true)
    y_p = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(y_t * y_p)
    return (2. * inter + smooth) / (tf.keras.backend.sum(y_t) + tf.keras.backend.sum(y_p) + smooth)

# %% ============================================================
# CÉLULA 4 — PREPARAÇÃO DE DADOS (com hard negatives de estradas)
# ============================================================
def preparar_dados():
    print("📊 Preparando dados...")
    gdf_estufas = gpd.read_file(SHP_ESTUFAS)
    print(f"  ✓ {len(gdf_estufas)} estufas de treino")

    with rasterio.open(TIFF_TREINO) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)   # RGB, descarta alpha se existir
        transform, crs = src.transform, src.crs
    print(f"  ✓ Imagem: {img.shape}")

    def rasterizar_shp(path):
        g = gpd.read_file(path)
        if g.crs != crs:
            g = g.to_crs(crs)
        return rasterize([(geom, 1) for geom in g.geometry],
                         out_shape=img.shape[:2], transform=transform,
                         fill=0, dtype='uint8')

    mascara_area = rasterizar_shp(SHP_MASCARA)
    if gdf_estufas.crs != crs:
        gdf_estufas = gdf_estufas.to_crs(crs)
    mask_estufas = rasterize([(geom, 1) for geom in gdf_estufas.geometry],
                             out_shape=img.shape[:2], transform=transform,
                             fill=0, dtype='uint8') * mascara_area
    print(f"  ✓ Pixels de estufa: {int(mask_estufas.sum()):,}")

    ps = PATCH_SIZE
    stride = ps // 2
    patches_img, patches_mask = [], []

    # 1) Patches centrados em estufas
    contours, _ = cv2.findContours(mask_estufas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            for dx in (-stride//2, 0, stride//2):
                for dy in (-stride//2, 0, stride//2):
                    x, y = cx + dx - ps//2, cy + dy - ps//2
                    if 0 <= x and 0 <= y and x+ps <= img.shape[1] and y+ps <= img.shape[0]:
                        pm = mask_estufas[y:y+ps, x:x+ps]
                        if pm.sum() >= MIN_ESTUFA_PIXELS:
                            patches_img.append(img[y:y+ps, x:x+ps])
                            patches_mask.append(pm)
    n_pos = len(patches_img)
    print(f"  ✓ {n_pos} patches positivos (centrados em estufas)")

    # 2) Hard negatives
    n_neg_alvo = int(n_pos * FRACAO_NEGATIVOS)
    n_estradas = 0

    def patch_negativo_valido(y, x):
        if mascara_area[y:y+ps, x:x+ps].sum() < ps*ps*0.7:
            return False
        if mask_estufas[y:y+ps, x:x+ps].sum() > 0:
            return False
        p = img[y:y+ps, x:x+ps]
        return p.mean() > 10 and p.std() > 5

    if USAR_NEGATIVOS_ESTRADAS and os.path.exists(SHP_ESTRADAS):
        mask_estradas = rasterizar_shp(SHP_ESTRADAS) * mascara_area
        ys, xs = np.where(mask_estradas == 1)
        print(f"  Pixels de estrada disponíveis: {len(ys):,}")
        if len(ys) > 0:
            idx = np.random.permutation(len(ys))
            alvo_estradas = n_neg_alvo // 2          # 50% dos negativos em estradas
            for k in idx:
                y = int(ys[k]) - ps//2
                x = int(xs[k]) - ps//2
                if 0 <= y <= img.shape[0]-ps and 0 <= x <= img.shape[1]-ps and patch_negativo_valido(y, x):
                    patches_img.append(img[y:y+ps, x:x+ps])
                    patches_mask.append(np.zeros((ps, ps), dtype='uint8'))
                    n_estradas += 1
                    if n_estradas >= alvo_estradas:
                        break
        print(f"  ✓ {n_estradas} hard negatives de ESTRADAS")

    n_aleatorios = 0
    for _ in range((n_neg_alvo - n_estradas) * 20):
        y = np.random.randint(0, img.shape[0] - ps)
        x = np.random.randint(0, img.shape[1] - ps)
        if patch_negativo_valido(y, x):
            patches_img.append(img[y:y+ps, x:x+ps])
            patches_mask.append(np.zeros((ps, ps), dtype='uint8'))
            n_aleatorios += 1
            if n_estradas + n_aleatorios >= n_neg_alvo:
                break
    print(f"  ✓ {n_aleatorios} negativos aleatórios")
    print(f"  ✓ Total: {len(patches_img)} patches")

    return (np.array(patches_img, dtype='uint8'),
            np.array(patches_mask, dtype='uint8'))

X, y = preparar_dados()

# %% ============================================================
# CÉLULA 5 — SPLIT + GERADOR DE AUGMENTATION (RAM constante)
# ============================================================
# Os patches originais ficam em uint8 (4x menos RAM que float32). A
# normalização /255 e a augmentation são feitas patch-a-patch DURANTE o
# treino, por um gerador — a memória NÃO cresce com N_AUGMENT.
X_train_u8, X_val_u8, y_train_u8, y_val_u8 = train_test_split(
    X, y, test_size=0.2, random_state=SEED,
    stratify=(y.sum(axis=(1, 2)) > 0)
)
del X, y

# Validação: pequena, pode ir toda para float32 de uma vez (sem augmentation)
X_val = X_val_u8.astype('float32') / 255.0
y_val = np.expand_dims(y_val_u8.astype('float32'), -1)

def augment(img, mask):
    if np.random.random() > 0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    if np.random.random() > 0.5:
        img, mask = np.flipud(img), np.flipud(mask)
    k = np.random.randint(0, 4)
    if k:
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    if np.random.random() > 0.5:
        img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
    if np.random.random() > 0.5:
        m = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - m) * np.random.uniform(0.8, 1.2) + m, 0, 1)
    if np.random.random() > 0.7:
        img = np.clip(img + np.random.normal(0, 0.01, img.shape), 0, 1)
    return img.astype('float32'), mask.astype('float32')

class GeradorEstufas(keras.utils.Sequence):
    """Serve batches normalizados; aplica augmentation ao vivo se aug=True.
    RAM ocupada = só os patches uint8 + um batch de cada vez."""
    def __init__(self, X_u8, y_u8, batch_size, aug):
        self.X, self.y = X_u8, y_u8
        self.bs, self.aug = batch_size, aug
        self.idx = np.arange(len(X_u8))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def on_epoch_end(self):
        if self.aug:
            np.random.shuffle(self.idx)

    def __getitem__(self, b):
        ids = self.idx[b*self.bs:(b+1)*self.bs]
        bx, by = [], []
        for i in ids:
            im = self.X[i].astype('float32') / 255.0
            mk = self.y[i].astype('float32')
            if self.aug and np.random.random() < (N_AUGMENT / (N_AUGMENT + 1) if N_AUGMENT else 0):
                im, mk = augment(im, mk)
            bx.append(im); by.append(mk)
        return np.array(bx), np.expand_dims(np.array(by), -1)

gen_train = GeradorEstufas(X_train_u8, y_train_u8, BATCH_SIZE, aug=True)
print(f"Treino: {len(X_train_u8)} patches (augmentation ao vivo) | "
      f"Validação: {len(X_val)}")
print(f"RAM: patches mantidos em uint8, normalização e augmentation por batch")

# %% ============================================================
# CÉLULA 6 — MODELO
# ============================================================
def unet_classico(input_shape=(256, 256, 3)):
    """A arquitetura U-Net do ensaio de 2025 (baseline)."""
    inputs = keras.Input(shape=input_shape)
    def bloco(x, f, drop):
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return layers.Dropout(drop)(x)
    c1 = bloco(inputs, 64, 0.1); p1 = layers.MaxPooling2D()(c1)
    c2 = bloco(p1, 128, 0.1);    p2 = layers.MaxPooling2D()(c2)
    c3 = bloco(p2, 256, 0.2);    p3 = layers.MaxPooling2D()(c3)
    c4 = bloco(p3, 512, 0.2);    p4 = layers.MaxPooling2D()(c4)
    c5 = bloco(p4, 1024, 0.3)
    def sobe(x, skip, f, drop):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        x = layers.concatenate([x, skip])
        return bloco(x, f, drop)
    c6 = sobe(c5, c4, 512, 0.2)
    c7 = sobe(c6, c3, 256, 0.2)
    c8 = sobe(c7, c2, 128, 0.1)
    c9 = sobe(c8, c1, 64, 0.1)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    return keras.Model(inputs, outputs)

if ARQUITETURA == 'unet':
    model = unet_classico((PATCH_SIZE, PATCH_SIZE, 3))
else:
    import segmentation_models as sm
    # NOTA DE COERÊNCIA: mantemos a normalização /255 em todo o pipeline (treino
    # e inferência), tal como no E0. O encoder ImageNet foi pré-treinado com outra
    # normalização, mas com fine-tuning (lr baixo, muitas épocas) adapta-se bem a
    # /255. Manter /255 preserva a REGRA DE OURO (treino=inferência) e evita ter de
    # replicar o pré-processamento do EfficientNet na função detetar(). Se algum dia
    # se quiser o pré-processamento nativo, teria de ser aplicado NOS DOIS lados.
    model = sm.Unet(ARQUITETURA, input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
                    classes=1, activation='sigmoid', encoder_weights='imagenet')

model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
              loss=combined_loss, metrics=[dice_coefficient])
print(f"Modelo {ARQUITETURA}: {model.count_params():,} parâmetros")

# %% ============================================================
# CÉLULA 7 — TREINO
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = DATA + f'modelo_{EXPERIMENTO}_{timestamp}.h5'   # gravado DIRETAMENTE no Drive

callbacks = [
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_dice_coefficient',
                                    mode='max', save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coefficient', mode='max',
                                      factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_dice_coefficient', mode='max',
                                  patience=25, restore_best_weights=True, verbose=1),
]

history = model.fit(gen_train, epochs=EPOCHS,
                    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

val_loss, val_dice = model.evaluate(X_val, y_val, verbose=0)
print(f"\n🎯 Dice validação: {val_dice:.3f}")

# Melhor threshold na validação
y_pred_val = model.predict(X_val, verbose=0)
best_threshold, best_dice = 0.5, 0
for t in np.arange(0.2, 0.85, 0.05):
    yb = (y_pred_val > t).astype('float32')
    inter = (y_val * yb).sum()
    d = 2 * inter / (y_val.sum() + yb.sum() + 1e-6)
    if d > best_dice:
        best_dice, best_threshold = d, float(t)
print(f"✓ Melhor threshold: {best_threshold:.2f} (dice {best_dice:.3f})")

metadata = {
    'experimento': EXPERIMENTO, 'arquitetura': ARQUITETURA,
    'normalizacao': 'div255', 'bandas': 'RGB(1,2,3)',
    'patch_size': PATCH_SIZE,
    'tversky': {'alpha': TVERSKY_ALPHA, 'beta': TVERSKY_BETA, 'gamma': TVERSKY_GAMMA},
    'negativos_estradas': USAR_NEGATIVOS_ESTRADAS,
    'best_threshold': best_threshold, 'val_dice': float(val_dice),
    'n_train': len(X_train_u8), 'n_val': len(X_val),
    'seed': SEED, 'timestamp': timestamp, 'model_path': MODEL_PATH,
}
with open(DATA + f'metadata_{EXPERIMENTO}_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Modelo e metadados gravados no Drive")

# %% ============================================================
# CÉLULA 8 — INFERÊNCIA NA MÁSCARA DE TESTE (pipeline coerente)
# ============================================================
def detetar(tiff_path, mask_shp, model, threshold):
    mask_gdf = gpd.read_file(mask_shp)
    with rasterio.open(tiff_path) as src:
        if mask_gdf.crs != src.crs:
            mask_gdf = mask_gdf.to_crs(src.crs)
        b = mask_gdf.total_bounds
        # BUG CORRIGIDO (2025): src.index(x,y) devolve (row, col), não (col, row).
        # Desempacotar ao contrário transpunha a janela de leitura.
        row_min, col_min = src.index(b[0], b[3])   # (row, col)!
        row_max, col_max = src.index(b[2], b[1])
        col_min, row_min = max(0, col_min), max(0, row_min)
        col_max, row_max = min(src.width, col_max), min(src.height, row_max)
        win = Window(col_min, row_min, col_max - col_min, row_max - row_min)
        img = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
        transform = src.window_transform(win)
        crs = src.crs

    img_norm = img.astype('float32') / 255.0       # MESMA normalização do treino
    h, w = img_norm.shape[:2]
    ps, stride = PATCH_SIZE, PATCH_SIZE - 64
    pred = np.zeros((h, w), 'float32'); cnt = np.zeros((h, w), 'float32')

    lote, coords = [], []
    def processa_lote():
        if not lote:
            return
        preds = model.predict(np.array(lote), verbose=0)
        for p, (yy, xx) in zip(preds, coords):
            ye, xe = min(yy+ps, h), min(xx+ps, w)
            pred[yy:ye, xx:xe] += p[:ye-yy, :xe-xx, 0]
            cnt[yy:ye, xx:xe] += 1
        lote.clear(); coords.clear()

    for i in range(0, max(1, h - ps + 1), stride):
        for j in range(0, max(1, w - ps + 1), stride):
            patch = img_norm[i:min(i+ps, h), j:min(j+ps, w)]
            if patch.shape[:2] != (ps, ps):
                pad = np.zeros((ps, ps, 3), 'float32')
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            lote.append(patch); coords.append((i, j))
            if len(lote) == 16:
                processa_lote()
    processa_lote()
    cnt[cnt == 0] = 1
    pred /= cnt

    # Recorte ao polígono da máscara
    # BUG CORRIGIDO (2025): a janela lida é o retângulo envolvente da máscara
    # (maior que o polígono real). Aqui rasteriza-se o polígono e anula-se tudo
    # o que fica fora dele, para não haver deteções fora da área de interesse.
    poligono = rasterize([(g, 1) for g in mask_gdf.geometry], out_shape=(h, w),
                         transform=transform, fill=0, dtype='uint8')
    pred *= poligono

    binm = (pred > threshold).astype('uint8')
    k = np.ones((3, 3), 'uint8')
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys, attrs = [], []
    rej = {'small': 0, 'large': 0, 'road_like': 0}
    for c in contours:
        if len(c) < 3:
            continue
        coords_xy = [transform * (pt[0][0], pt[0][1]) for pt in c]
        try:
            poly = Polygon(coords_xy).simplify(1.0, preserve_topology=True)
        except Exception:
            continue
        a = poly.area
        if a < MIN_AREA_M2:
            rej['small'] += 1; continue
        if a > MAX_AREA_M2:
            rej['large'] += 1; continue
        compac = 4 * np.pi * a / max(poly.length**2, 1e-6)
        try:
            cs = list(poly.minimum_rotated_rectangle.exterior.coords)
            lados = [np.hypot(cs[q+1][0]-cs[q][0], cs[q+1][1]-cs[q][1]) for q in range(4)]
            along = max(lados[0], lados[1]) / max(min(lados[0], lados[1]), 0.1)
        except Exception:
            along = 1.0
        if compac < MIN_COMPACIDADE or along > MAX_ALONGAMENTO:
            rej['road_like'] += 1; continue
        polys.append(poly); attrs.append((compac, along))

    print(f"✓ {len(polys)} deteções | rejeitadas: {rej}")
    gdf = gpd.GeoDataFrame({'id': range(len(polys)),
                            'area_m2': [p.area for p in polys],
                            'compac': [a[0] for a in attrs],
                            'along': [a[1] for a in attrs],
                            'geometry': polys}, crs=crs)
    return gdf

thr = CONFIDENCE_THRESHOLD if CONFIDENCE_THRESHOLD else best_threshold
print(f"Inferência com threshold {thr:.2f}...")
gdf_pred = detetar(TIFF_TESTE, SHP_MASC_TESTE, model, thr)
out_shp = DATA + f'resultado_{EXPERIMENTO}_{timestamp}.shp'
gdf_pred.to_file(out_shp)
print(f"✓ Shapefile: {out_shp}")

# %% ============================================================
# CÉLULA 9 — AVALIAÇÃO CONTRA A VERDADE DE TERRENO
# ============================================================
def avaliar(gdf_pred, gdf_gt, iou_min=0.3):
    gt_usadas, tp = set(), 0
    for _, pr in gdf_pred.iterrows():
        for idx, gt in gdf_gt.iterrows():
            if idx in gt_usadas:
                continue
            inter = pr.geometry.intersection(gt.geometry).area
            union = pr.geometry.union(gt.geometry).area
            if union > 0 and inter / union >= iou_min:
                tp += 1; gt_usadas.add(idx); break
    fp, fn = len(gdf_pred) - tp, len(gdf_gt) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return dict(TP=tp, FP=fp, FN=fn, precisao=round(prec, 3),
                recall=round(rec, 3), F1=round(f1, 3))

gdf_gt = gpd.read_file(SHP_VERDADE)
if gdf_gt.crs != gdf_pred.crs:
    gdf_gt = gdf_gt.to_crs(gdf_pred.crs)

# RECORTE CRÍTICO: manter só as estufas cujo centroide cai dentro da máscara de
# teste. Sem isto, estufas reais fora da máscara (onde o modelo nem processou)
# contariam como falsos negativos e afundariam o recall artificialmente.
_mask = gpd.read_file(SHP_MASC_TESTE)
if _mask.crs != gdf_pred.crs:
    _mask = _mask.to_crs(gdf_pred.crs)
_mask_union = _mask.union_all() if hasattr(_mask, 'union_all') else _mask.unary_union
_n_antes = len(gdf_gt)
gdf_gt = gdf_gt[gdf_gt.geometry.centroid.within(_mask_union)].reset_index(drop=True)
print(f"Verdade de terreno: {_n_antes} total -> {len(gdf_gt)} dentro da máscara")

res = avaliar(gdf_pred, gdf_gt)
print(f"\n{'='*60}\nRESULTADO {EXPERIMENTO} (threshold {thr:.2f})")
print(f"  Deteções: {len(gdf_pred)} | Verdade de terreno: {len(gdf_gt)}")
for k, v in res.items():
    print(f"  {k}: {v}")
print('='*60)

metadata['avaliacao'] = res
with open(DATA + f'metadata_{EXPERIMENTO}_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Avaliação acrescentada aos metadados — linha pronta para a tabela comparativa")
