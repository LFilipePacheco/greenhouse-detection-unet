import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

# ========================================
# CONFIGURAÇÕES MELHORADAS PARA RETREINO
# ========================================

# Configurar GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"🎮 GPU disponível: {physical_devices[0]}")

# ========================================
# FUNÇÕES DE LOSS OTIMIZADAS
# ========================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss para segmentação"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def focal_tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=1.0):
    """
    Loss especializada - AJUSTADA para melhor balance
    alpha=0.5, beta=0.5 para balance entre FP e FN
    """
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    true_pos = tf.keras.backend.sum(y_true_f * y_pred_f)
    false_neg = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    focal_tversky = tf.keras.backend.pow((1 - tversky), gamma)
    
    return focal_tversky

def combined_loss(y_true, y_pred):
    """Combinação com mais peso no Dice"""
    return 0.7 * dice_loss(y_true, y_pred) + 0.3 * focal_tversky_loss(y_true, y_pred)

def dice_coefficient(y_true, y_pred):
    """Métrica Dice"""
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

# ========================================
# DATA AUGMENTATION MELHORADA
# ========================================

def augment_batch(images, masks):
    """
    Augmentation mais agressiva para melhor generalização
    """
    augmented_images = []
    augmented_masks = []
    
    for img, mask in zip(images, masks):
        # Sempre incluir original
        augmented_images.append(img)
        augmented_masks.append(mask)
        
        # Aumentar variabilidade
        for _ in range(3):  # 3 augmentações por imagem
            aug_img = img.copy()
            aug_mask = mask.copy()
            
            # Random flip
            if np.random.random() > 0.5:
                aug_img = np.fliplr(aug_img)
                aug_mask = np.fliplr(aug_mask)
            
            if np.random.random() > 0.5:
                aug_img = np.flipud(aug_img)
                aug_mask = np.flipud(aug_mask)
            
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                aug_img = np.rot90(aug_img, k)
                aug_mask = np.rot90(aug_mask, k)
            
            # Ajuste de brilho/contraste
            if np.random.random() > 0.5:
                # Brilho
                factor = np.random.uniform(0.7, 1.3)
                aug_img = aug_img * factor
                aug_img = np.clip(aug_img, 0, 1)
            
            if np.random.random() > 0.5:
                # Contraste
                factor = np.random.uniform(0.8, 1.2)
                mean = np.mean(aug_img, axis=(0, 1), keepdims=True)
                aug_img = (aug_img - mean) * factor + mean
                aug_img = np.clip(aug_img, 0, 1)
            
            # Adicionar ruído
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.01, aug_img.shape)
                aug_img = aug_img + noise
                aug_img = np.clip(aug_img, 0, 1)
            
            augmented_images.append(aug_img)
            augmented_masks.append(aug_mask)
    
    return np.array(augmented_images), np.array(augmented_masks)

# ========================================
# PREPARAÇÃO DE DADOS MELHORADA
# ========================================

def preparar_dados_melhorado_v2(
    shapefile_path,
    tiff_path,
    mascara_path,
    patch_size=256,
    overlap=0.5,  # AUMENTADO de 0.3 para 0.5
    min_estufa_pixels=100  # Mínimo de pixels de estufa no patch
):
    """
    Preparação com foco em patches de qualidade
    """
    print("📊 Preparando dados - Versão Melhorada...")
    
    # Carregar shapefile das estufas
    print("  Carregando estufas...")
    gdf_estufas = gpd.read_file(shapefile_path)
    print(f"  ✓ {len(gdf_estufas)} estufas carregadas")
    
    # Carregar imagem
    print("  Carregando imagem...")
    with rasterio.open(tiff_path) as src:
        # Verificar número de bandas
        if src.count >= 3:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
        else:
            raise ValueError(f"Imagem tem apenas {src.count} bandas, precisa de RGB (3)")
        
        transform = src.transform
        crs = src.crs
    
    print(f"  ✓ Imagem: {img.shape}")
    
    # Carregar e processar máscara
    print("  Processando máscara...")
    if mascara_path.endswith('.shp'):
        gdf_mascara = gpd.read_file(mascara_path)
        if gdf_mascara.crs != crs:
            gdf_mascara = gdf_mascara.to_crs(crs)
        
        mascara_area = rasterize(
            [(geom, 1) for geom in gdf_mascara.geometry],
            out_shape=img.shape[:2],
            transform=transform,
            fill=0,
            dtype='uint8'
        )
    else:
        with rasterio.open(mascara_path) as src_mask:
            mascara_area = src_mask.read(1) > 0
    
    # Criar máscara de estufas
    print("  Criando máscara de estufas...")
    mask_estufas = rasterize(
        [(geom, 1) for geom in gdf_estufas.geometry],
        out_shape=img.shape[:2],
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    
    # Aplicar máscara de área
    mask_estufas = mask_estufas * mascara_area
    
    print(f"  ✓ Pixels de estufa: {np.sum(mask_estufas):,}")
    
    # Extrair patches com estratégia melhorada
    patches_img = []
    patches_mask = []
    
    stride = int(patch_size * (1 - overlap))
    
    # ESTRATÉGIA 1: Patches centrados em estufas (PRIORIDADE)
    print("  Extraindo patches centrados em estufas...")
    
    # Encontrar centros de estufas
    contours, _ = cv2.findContours(mask_estufas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches_por_estufa = 5  # Mais patches por estufa
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Extrair múltiplos patches ao redor do centro
            for dx in [-stride//2, 0, stride//2]:
                for dy in [-stride//2, 0, stride//2]:
                    x = cx + dx - patch_size // 2
                    y = cy + dy - patch_size // 2
                    
                    # Verificar bounds
                    if x >= 0 and y >= 0 and x + patch_size <= img.shape[1] and y + patch_size <= img.shape[0]:
                        patch_img = img[y:y+patch_size, x:x+patch_size]
                        patch_mask = mask_estufas[y:y+patch_size, x:x+patch_size]
                        
                        # Só incluir se tiver pixels de estufa suficientes
                        if np.sum(patch_mask) >= min_estufa_pixels:
                            patches_img.append(patch_img)
                            patches_mask.append(patch_mask)
    
    print(f"  ✓ {len(patches_img)} patches centrados em estufas")
    
    # ESTRATÉGIA 2: Patches com sliding window (complementar)
    print("  Extraindo patches adicionais com sliding window...")
    patches_adicionais = 0
    
    for i in range(0, img.shape[0] - patch_size, stride):
        for j in range(0, img.shape[1] - patch_size, stride):
            patch_mascara_area = mascara_area[i:i+patch_size, j:j+patch_size]
            
            # Verificar se está na área de interesse
            if np.sum(patch_mascara_area) < (patch_size * patch_size * 0.5):
                continue
            
            patch_mask = mask_estufas[i:i+patch_size, j:j+patch_size]
            
            # Incluir patches com estufas
            if np.sum(patch_mask) >= min_estufa_pixels:
                patch_img = img[i:i+patch_size, j:j+patch_size]
                
                # Evitar duplicatas (verificar se já não foi adicionado)
                if not any(np.array_equal(patch_img, p) for p in patches_img[-20:]):
                    patches_img.append(patch_img)
                    patches_mask.append(patch_mask)
                    patches_adicionais += 1
    
    print(f"  ✓ {patches_adicionais} patches adicionais")
    
    # ESTRATÉGIA 3: Hard negatives (patches sem estufas mas similares)
    print("  Adicionando hard negatives...")
    num_negativos = len(patches_img) // 3  # 33% de negativos
    negativos_adicionados = 0
    
    for _ in range(num_negativos * 10):  # Tentativas
        i = np.random.randint(0, img.shape[0] - patch_size)
        j = np.random.randint(0, img.shape[1] - patch_size)
        
        patch_mascara_area = mascara_area[i:i+patch_size, j:j+patch_size]
        if np.sum(patch_mascara_area) < (patch_size * patch_size * 0.7):
            continue
        
        patch_mask = mask_estufas[i:i+patch_size, j:j+patch_size]
        
        # Patches completamente sem estufas
        if np.sum(patch_mask) == 0:
            patch_img = img[i:i+patch_size, j:j+patch_size]
            
            # Verificar se tem conteúdo (não é preto/vazio)
            if np.mean(patch_img) > 10 and np.std(patch_img) > 5:
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
                negativos_adicionados += 1
                
                if negativos_adicionados >= num_negativos:
                    break
    
    print(f"  ✓ {negativos_adicionados} hard negatives")
    print(f"  ✓ Total: {len(patches_img)} patches")
    
    return np.array(patches_img), np.array(patches_mask)

# ========================================
# MODELO U-NET MELHORADO
# ========================================

def unet_melhorado(input_shape=(256, 256, 3)):
    """
    U-Net com melhorias: Dropout, BatchNorm, mais filtros
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder com mais capacidade
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)  # 32->64
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.1)(c1)  # Dropout leve
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)  # 64->128
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.1)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)  # 128->256
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)  # 256->512
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.2)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)  # 512->1024
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)
    
    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.2)(c6)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.2)(c7)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Dropout(0.1)(c8)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Dropout(0.1)(c9)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# ========================================
# TREINO PRINCIPAL MELHORADO
# ========================================

def treinar_modelo_melhorado():
    """
    Treino otimizado para melhor Dice Score
    """
    
    # Configurações
    SHAPEFILE = r"path/to/validated_greenhouses.shp"   # ground-truth polygons (GIS-validated)
    TIFF = r"path/to/sentinel2_composite.tif"          # multiband Sentinel-2 L2A composite
    MASCARA = r"path/to/training_area_mask.shp"        # area of interest for sampling
    OUTPUT_DIR = r"path/to/output"
    
    print("="*70)
    print("🚀 RETREINO OTIMIZADO - OBJETIVO: DICE > 0.75")
    print("="*70)
    
    # 1. Preparar dados
    X, y = preparar_dados_melhorado_v2(
        SHAPEFILE, TIFF, MASCARA,
        patch_size=256,
        overlap=0.5,  # Maior overlap
        min_estufa_pixels=100
    )
    
    if len(X) == 0:
        print("❌ Nenhum patch extraído!")
        return None, None
    
    # Normalizar
    X = X.astype('float32') / 255.0
    y = y.astype('float32')
    y = np.expand_dims(y, axis=-1)
    
    # Split 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(np.sum(y, axis=(1,2,3)) > 0)
    )
    
    # Aplicar augmentation ao treino
    print("\n🔄 Aplicando data augmentation...")
    X_train_aug, y_train_aug = augment_batch(X_train, y_train)
    
    print(f"\n📊 Dataset final:")
    print(f"  Treino original: {len(X_train)} patches")
    print(f"  Treino com augmentation: {len(X_train_aug)} patches")
    print(f"  Validação: {len(X_val)} patches")
    
    # 2. Criar modelo melhorado
    print("\n🏗️ Construindo modelo melhorado...")
    model = unet_melhorado(input_shape=(256, 256, 3))
    print(f"  ✓ Modelo com {model.count_params():,} parâmetros")
    
    # 3. Compilar com learning rate menor
    initial_lr = 0.0005  # Menor que 0.001
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    # 4. Callbacks melhorados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(OUTPUT_DIR, f'modelo_melhorado_{timestamp}.h5')
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=30,  # Mais paciência
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, f'logs_{timestamp}'),
            histogram_freq=1
        )
    ]
    
    # 5. Treinar
    print("\n🚀 Iniciando treino melhorado...")
    print("  Objetivo: Dice Score > 0.75")
    print("  Isto pode demorar 30-60 minutos...")
    
    history = model.fit(
        X_train_aug, y_train_aug,
        batch_size=8,
        epochs=150,  # Mais epochs
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Avaliar
    print("\n📊 Avaliação final:")
    val_loss, val_acc, val_dice = model.evaluate(X_val, y_val, verbose=0)
    print(f"  🎯 Dice Score: {val_dice:.3f}")
    print(f"  📊 Accuracy: {val_acc:.3f}")
    
    # 7. Encontrar melhor threshold
    print("\n🔍 Procurando melhor threshold...")
    y_pred = model.predict(X_val[:100], verbose=0)  # Usar subset para velocidade
    
    best_threshold = 0.5
    best_dice = 0
    
    for threshold in np.arange(0.2, 0.8, 0.05):
        y_pred_binary = (y_pred > threshold).astype(np.float32)
        
        intersection = np.sum(y_val[:100] * y_pred_binary)
        dice = (2. * intersection) / (np.sum(y_val[:100]) + np.sum(y_pred_binary) + 1e-6)
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    print(f"  ✓ Melhor threshold: {best_threshold:.2f}")
    
    # 8. Salvar metadados
    metadata = {
        'best_threshold': float(best_threshold),
        'final_dice': float(val_dice),
        'final_loss': float(val_loss),
        'final_accuracy': float(val_acc),
        'epochs_trained': len(history.history['loss']),
        'best_epoch': np.argmax(history.history['val_dice_coefficient']) + 1,
        'patch_size': 256,
        'overlap': 0.5,
        'num_train_patches': len(X_train_aug),
        'num_val_patches': len(X_val),
        'timestamp': timestamp
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f'modelo_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Modelo salvo: {model_path}")
    print(f"📄 Metadados: {metadata_path}")
    
    # 9. Visualização
    plt.figure(figsize=(15, 5))
    
    # Dice Score
    plt.subplot(1, 3, 1)
    plt.plot(history.history['dice_coefficient'], label='Treino', linewidth=2)
    plt.plot(history.history['val_dice_coefficient'], label='Validação', linewidth=2)
    plt.title('Dice Score', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True, alpha=0.3)
    best_dice_epoch = np.argmax(history.history['val_dice_coefficient'])
    plt.axvline(x=best_dice_epoch, color='red', linestyle='--', alpha=0.5)
    plt.text(best_dice_epoch, 0.5, f'Best: {val_dice:.3f}', rotation=90)
    
    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Treino', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
    plt.title('Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning Rate
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], linewidth=2, color='green')
        plt.title('Learning Rate', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'TREINO MELHORADO - Dice Final: {val_dice:.3f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, f'training_curves_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Gráficos salvos: {fig_path}")
    
    # Análise de melhoria
    print("\n" + "="*70)
    if val_dice > 0.75:
        print("🎉 OBJETIVO ALCANÇADO! Dice > 0.75")
    elif val_dice > 0.65:
        print("✅ BOM RESULTADO! Dice > 0.65")
    else:
        print("⚠️ Resultado abaixo do esperado. Sugestões:")
        print("  - Adicionar mais dados de treino")
        print("  - Marcar manualmente mais estufas")
        print("  - Ajustar hiperparâmetros")
    
    print("="*70)
    
    return model, metadata

# ========================================
# EXECUÇÃO
# ========================================

if __name__ == "__main__":
    print("🔧 Iniciando retreino melhorado...")
    print("Objetivo: Reduzir falsos positivos e detectar estufas perdidas")
    
    model, metadata = treinar_modelo_melhorado()
    
    if metadata:
        print(f"\n✅ RETREINO CONCLUÍDO!")
        print(f"🎯 Dice Score final: {metadata['final_dice']:.3f}")
        print(f"🎯 Melhor threshold: {metadata['best_threshold']:.2f}")
        print("\n📝 Próximos passos:")
        print(f"1. Use o novo modelo para detecção")
        print(f"2. Use threshold = {metadata['best_threshold']:.2f}")
        print(f"3. Se ainda houver problemas, adicione mais dados de treino")