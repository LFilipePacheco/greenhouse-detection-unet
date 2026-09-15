# Scripts — pipeline de deteção de estufas

Implementação de referência do pipeline treino-e-deteção. Os caminhos no topo de
cada script são **exemplos** — ajuste-os ao seu ambiente (pastas `data/`,
`models/`, `output/`).

## Ordem de utilização

**1. Preparação de dados**
- `recortar_fase0.py` — recorta a ortoimagem às máscaras de treino/teste
  (compressão LZW), produzindo os GeoTIFF de trabalho.

**2. Treino** (Google Colab, GPU T4)
- `treino_estufas_colab_comentado.py` — treino do modelo E0 (U-Net). Versão
  comentada, de referência.
- `treino_estufas_E2_colab.py` — treino do modelo E2 (U-Net + encoder ResNet34,
  ImageNet).

**3. Deteção pontual** (local, CPU)
- `deteta_estufas_DL_completo_v2.py` — aplica o modelo a uma máscara/área única
  (sliding window). Inclui as correções de pré-processamento (normalização /255,
  recorte ao polígono).

**4. Produção à escala da Zona Vulnerável** (local, CPU, retomável)
- `producao_1_gerar_grelha.py` — gera a grelha de tiles que cobre a zona.
- `producao_2_inferir_tiles.py` — inferência em lote sobre todos os tiles +
  consolidação. Filtro de área: mínimo 40 m², **sem teto máximo**.

**5. Avaliação**
- `avaliar_detecao.py` — avaliação por objeto (IoU ≥ 0.3), com recorte da verdade
  de terreno à máscara; gera camadas de diagnóstico (TP/FP e FN).
- `avaliar_por_area.py` — avaliação por área (precisão/recall/F1 de área, IoU
  global), robusta à fragmentação de polígonos.
- `posproc_morfologico.py` — teste de pós-processamento morfológico
  (preenchimento de buracos + fecho). Documentado como beco sem saída: sem efeito
  no recall de área depois de removido o teto de área.

## Parâmetros do modelo final (E0)

Normalização `/255` (treino e inferência) · patch 256×256 · stride 192 ·
limiar 0.4 · filtro geométrico desligado · **sem teto de área** (mín. 40 m²) ·
loss 0.7·Dice + 0.3·FocalTversky · Adam lr=5e-4.

Unidade de deteção: **bloco contíguo de cobertura plástica**.

## Dependências

Python · TensorFlow/Keras · rasterio · GeoPandas · OpenCV · Shapely ·
scikit-learn · (E2: segmentation-models)

## Nota sobre dados

Os dados (ortoimagem, pesos do modelo, verdade de terreno) não estão incluídos.
A ortoimagem ortoSat2023 é um serviço aberto da Direção-Geral do Território (DGT).
Os restantes dados são propriedade institucional da CCDR-Norte.
