🇵🇹 **[English](README.md)**

# Deteção de estufas em imagens de satélite com aprendizagem profunda
### Segmentação semântica com U-Net · Zona Vulnerável de Esposende – Vila do Conde, Portugal

> Pipeline reprodutível de treino e deteção para cartografar estruturas de estufas
> numa Zona Vulnerável ao abrigo da Diretiva Nitratos (~20 000 ha), a partir de
> ortoimagens de satélite de alta resolução (DGT *ortoSat2023*, RGB, ~30 cm/píxel).
> Desenvolvido com base no diagnóstico de erros, numa separação rigorosa entre
> treino e teste e numa verdade de terreno criteriosa, bem como no princípio de
> desconfiar dos próprios resultados até que cada conclusão resista a uma análise
> crítica. O modelo final permite **localizar, contabilizar e medir de forma fiável
> a área das estufas**.

---

![Estufas detetadas sobre ortoimagem](docs/detections_overlay.png)
*Polígonos de estufas detetadas (amarelo) sobre ortoimagem de alta resolução: resultado do modelo final após validação em SIG.*

## Resultado principal

Avaliação realizada com base numa **verdade de terreno rigorosamente
vetorizada** (83 estufas numa subzona de ~50 ha), utilizando o modelo E0 sem
limite máximo de área:

| Tarefa | Precisão | Recall | F1 |
|---|---:|---:|---:|
| **Deteção por objeto** (localização/contagem) | **0,84** | **0,71** | **0,77** |
| **Cobertura de área** | **0,96** | **0,85** | **0,90** |

Ambos os objetivos foram atingidos: o modelo é fiável para **localizar e contar**
estufas e para **estimar a área** por elas ocupada. Na subzona, a área detetada é
de 9,1 ha, face a 10,3 ha de área real. A unidade de deteção é definida como o
**bloco contíguo de cobertura plástica**, independentemente do número de túneis
individuais que contenha, por ser esta a convenção relevante para fins de gestão
do território.

A precisão constitui o ponto forte mais consistente: 0,84 por objeto e 0,96 por
área, mantendo-se estável entre zonas. O problema histórico de falsos positivos
do projeto-piloto de 2025 foi resolvido.

## Porquê detetar estufas

A horticultura intensiva sob plástico é uma das principais fontes de
contaminação dos aquíferos por nitratos na Zona Vulnerável de Esposende – Vila
do Conde (ZV). Uma camada completa e atualizada de estufas é simultaneamente um
instrumento de controlo do cumprimento das normas e um elemento para a
cartografia do risco. O seu cruzamento com a ocupação do solo, as práticas
agrícolas e as concentrações de nitratos permite à entidade de monitorização
direcionar a fiscalização para os locais onde a pressão sobre as águas
subterrâneas é mais elevada.

Existem também razões para suspeitar de um sub-registo significativo das
parcelas com estufas no sistema oficial de identificação parcelar (iSIP, gerido
pelo IFAP). A produção de um **inventário validado de estufas para toda a zona**,
que permita quantificar essa diferença, constitui o objetivo operacional que o
modelo torna agora possível, após a expansão da inferência às 19 máscaras de
processamento (ver *Trabalho futuro*).

A cartografia manual a esta escala é impraticável: estão em causa milhares de
hectares distribuídos por três municípios, com um parque de estufas que se
altera de campanha para campanha. A tarefa exigia automatização.

## De 2025 a 2026: o que mudou

O projeto-piloto de 2025 produziu demasiados falsos positivos, nomeadamente em
estradas asfaltadas e coberturas de edifícios, exigindo uma limpeza manual
considerável. Em vez de aplicar correções pontuais, a campanha de 2026
diagnosticou as causas principais e reconstruiu o pipeline. Os dois problemas
com maior impacto foram:

1. **Normalização inconsistente entre treino e inferência**: a principal causa
   dos falsos positivos. O treino utilizava normalização por `/255`, enquanto a
   inferência aplicava um estiramento de contraste com percentis (p2–p98). Assim,
   o modelo recebia píxeis numa escala diferente e classificava campos inteiros
   como estufas. **Correção:** utilização de `/255` em todo o pipeline. Esta única
   alteração aumentou a precisão por objeto de **0,13 para 0,85** na máscara de
   teste.
2. **Filtro obsoleto de área máxima**: um limite `MAX_AREA_M2 = 3000`, introduzido
   em 2025 para conter as manchas de falsos positivos causadas pelo erro de
   normalização. Depois de corrigida a normalização, este limite tornou-se
   prejudicial, eliminando silenciosamente blocos contíguos de túneis, ou seja,
   estufas reais de grande dimensão. A sua remoção aumentou o recall de área de
   **0,44 para 0,85** (ver abaixo). Este erro afetou todas as avaliações
   intermédias até ser identificado.

A leitura do raster foi também corrigida para `src.read([1,2,3])` (RGB),
descartando o canal alfa que o serviço WMTS exporta por vezes como quarta banda.

## Método (pipeline corrigido)

**Regra de ouro: coerência do pré-processamento.** Normalização `/255` e bandas
RGB fixas no treino, validação e inferência, registadas em metadados JSON de cada
modelo. O treino e a inferência nunca podem divergir.

**Separação inviolável entre treino e teste.** Zonas de treino e teste sem
sobreposição espacial, verificada geometricamente. Sem esta separação, a
avaliação mede memorização e não capacidade de generalização.

**Verdade de terreno rigorosa.** Estufas vetorizadas manualmente sobre a
ortoimagem segundo um critério consistente. Avaliação por objeto com IoU ≥ 0,3 e
avaliação por área robusta à fragmentação dos polígonos.

**Preparação dos dados de treino.** Patches de 256×256: (a) centrados nas estufas,
(b) janela deslizante complementar e (c) negativos difíceis. Aumento de dados em
tempo real através de um gerador `keras.utils.Sequence`, mantendo os patches em
`uint8` e executando a normalização e o aumento por lote. Esta solução resolveu
o esgotamento de RAM causado pelo aumento de dados pré-calculado no Colab.

**Ambiente de execução.** Treino no Google Colab (GPU T4); inferência e avaliação
localmente (CPU, Windows 11, 32 GB de RAM, sem GPU NVIDIA). O raster de 19 GB foi
recortado por máscara, com compressão LZW, para respeitar o limite da versão
gratuita do Google Drive. O raster completo nunca foi carregado.

## Experiências e resultados

Foram treinados e comparados dois modelos:

- **E0 (modelo de referência retreinado)**: U-Net (~31 milhões de parâmetros),
  função de perda combinada (0,7·Dice + 0,3·Focal Tversky), Adam `lr=5e-4`, uma
  única zona de treino. `val_dice ≈ 0,94`.
- **E2 (encoder pré-treinado, orientado para o recall)**: U-Net com encoder
  **ResNet34** pré-treinado no ImageNet, função de perda orientada para o recall e
  treino **multizona** (~3× mais dados do que o E0).

Avaliação por objeto (IoU ≥ 0,3):

| Execução | Zona de teste | GT | Precisão | Recall | F1 | Nota |
|---|---|---:|---:|---:|---:|---|
| Referência 2025 | máscara 1 | 33 | 0,13 | 0,76 | 0,23 | antes da correção |
| E0 (sem filtro, limiar 0,4) | máscara 1 | 33 | 0,85 | 0,88 | 0,87 | amostra pequena |
| E0 | máscara 2 | 231 | 0,83 | 0,72 | 0,77 | intermédio, limite de área ativo |
| E2 (multizona) | 488 ha | 536 | 0,83 | 0,71 | 0,77 | intermédio, limite de área ativo |
| **E0: referência** | **subzona rigorosa** | **83** | **0,84** | **0,71** | **0,77** | **final, sem limite máximo** |

> As execuções intermédias foram avaliadas com o limite de área ainda ativo; o
> valor final de referência (E0, subzona rigorosa, sem limite máximo) é o
> apresentado no resultado principal. O F1 por objeto mantém-se estável (~0,77)
> nas zonas representativas. A remoção do limite aumentou sobretudo o **recall de
> área** (0,44 para 0,85), e não as métricas por objeto.

Observações principais:

- A normalização coerente por `/255` fez aumentar a precisão de **0,13 para 0,85**.
- As zonas maiores, com 231 e 536 estufas, fornecem o valor mais fiável para a
  avaliação por objeto: **F1 ≈ 0,77**.
- **E0 ≈ E2**: um encoder pré-treinado e três vezes mais dados não melhoraram o F1.

### O filtro de área: uma autocorreção

Um contraste inicial entre o recall por objeto (~0,71) e o recall de área (~0,44)
sugeria que o modelo localizava as estufas, mas cobria apenas metade da sua
superfície. A causa revelou-se estar no **pós-processamento, e não no modelo**: o
limite obsoleto `MAX_AREA_M2` eliminava todas as deteções acima de 3 000 m²,
precisamente os blocos contíguos de túneis correspondentes a estufas reais de
grande dimensão.

Nova quantificação na subzona rigorosa (E0, com e sem limite máximo):

| Métrica | Com limite de 3 000 m² | Sem limite máximo |
|---|---:|---:|
| Recall por objeto | 0,60 | **0,71** |
| Recall de área | 0,44 | **0,85** |
| Área detetada (ha) | 4,77 | **9,09** |
| Precisão por objeto | 0,82 | 0,84 |
| Precisão de área | 0,95 | 0,96 |

Nove estufas de grande dimensão, num total de 83, estavam a ser rejeitadas pelo
filtro. A remoção do limite aumentou a área detetada de 4,8 para 9,1 ha, face a
10,3 ha reais, **sem perda de precisão**, demonstrando que se tratava de deteções
legítimas. O défice de área anteriormente atribuído ao modelo era, em grande
medida, um **artefacto do filtro**.

### O papel da qualidade da verdade de terreno

O fator mais determinante para as métricas **não** foi a escolha do modelo, mas a
qualidade da verdade de terreno. A reutilização de polígonos irregulares herdados
da deteção de 2025 inflacionava a referência e fragmentava estufas compridas,
penalizando deteções corretas como falsos positivos. Numa avaliação intermédia
(modelo E2, ainda com o filtro de área ativo), a vetorização de uma verdade de
terreno rigorosa a partir do zero aumentou a precisão por objeto de **0,83 para
0,92** e a precisão de área de **0,88 para 0,97**, confirmando que a maioria dos
aparentes «falsos positivos» correspondia, na realidade, a deteções corretas.
Estes valores isolam o efeito da verdade de terreno; o resultado final de
referência corresponde ao E0, sem limite máximo, apresentado no resultado
principal.

## Abordagens testadas e rejeitadas

Documentar as abordagens sem sucesso também faz parte do resultado:

- **Limiar de confiança**: as probabilidades são bimodais; o seu ajuste teve
  pouco efeito.
- **Filtro geométrico anti-estrada**: rejeitou 16 estufas reais, porque os túneis
  estreitos são geometricamente idênticos a troços de estrada, enquanto os
  principais falsos positivos, coberturas de edifícios, eram *mais* compactos do
  que as estufas. O filtro foi desativado; o problema dos falsos positivos é
  espectral e não geométrico.
- **Arquitetura mais profunda/pré-treinada (E2)**: sem melhoria de F1 face ao E0.
- **Três vezes mais dados de treino (multizona)**: sem melhoria de F1.
- **Pós-processamento morfológico** (preenchimento de buracos e fecho): sem efeito
  na área, porque, no momento desse teste, o limite de área aplicado a montante
  já tinha eliminado as estufas de grande dimensão. Nenhuma operação a jusante
  poderia recuperá-las. Após a remoção do limite, o recall de área aumentou para
  ~0,85 sem necessidade de morfologia.

## Principais conclusões

- Os maiores ganhos resultaram da **higiene dos dados**, e não de uma arquitetura
  sofisticada: normalização coerente, separação rigorosa entre treino e teste e
  verdade de terreno de qualidade.
- **É essencial auditar o pós-processamento herdado.** Um único parâmetro obsoleto
  afetou silenciosamente todas as métricas intermédias e quase conduziu a uma
  conclusão científica errada: «a área é irredutivelmente subestimada». Não era.
- **A qualidade da verdade de terreno dominou** todas as escolhas de modelação.
- A arquitetura e o volume de dados nem sempre são a variável decisiva: E0 ≈ E2
  e três vezes mais dados produziram o mesmo F1.

## Limitações

- As métricas mais fiáveis baseiam-se em 231–536 estufas nas zonas maiores e em
  83 estufas na subzona rigorosa.
- O recall de área (0,85) foi medido apenas na subzona rigorosa e deve ser
  reproduzido noutra zona.
- Foi utilizada uma única fonte e época de imagem (*ortoSat2023*); a capacidade
  de generalização temporal e entre sensores não foi avaliada.

## Trabalho futuro

- Aplicar o modelo a toda a ZV (~20 000 ha), produzindo um **inventário validado
  de estufas para toda a zona**, como base para uma quantificação defensável do
  sub-registo face ao cadastro iSIP-IFAP.
- Aumentar a verdade de terreno rigorosa, abrangendo várias subzonas e pelo menos
  300 estufas, e reproduzir a métrica de área numa segunda subzona.
- Explorar imagens de maior resolução ou multiespectrais para avaliar o limite
  físico da assinatura espectral do plástico.

## Receita de reprodução (modelo final)

- Imagens: *ortoSat2023* (DGT), RGB, ~30 cm/píxel, EPSG:3763
- Normalização: `/255` no treino **e** na inferência
- Patch de 256×256; passo da janela deslizante de 192 (patch − 64)
- Limiar de inferência: **0,4**
- Filtro geométrico: **desativado**; filtro de área: **mínimo de 40 m², sem limite
  máximo** (o limite de 3 000 m² eliminava blocos contíguos de túneis)
- E0: U-Net com ~31 milhões de parâmetros, função de perda
  0,7·Dice + 0,3·FocalTversky(0,3/0,5/1,33), Adam 5e-4, lote 8 e paragem antecipada
  com base em `val_dice`
- E2: U-Net + ResNet34 (ImageNet), Tversky(0,6/0,4), Adam 1e-4
- Avaliação: por objeto (IoU ≥ 0,3) e por área; verdade de terreno recortada pela
  máscara
- Unidade de deteção: bloco contíguo de cobertura plástica

**Resultado de referência (E0, subzona rigorosa, 83 estufas, ~50 ha, sem limite
de área):** precisão por objeto = 0,84; recall = 0,71; F1 = 0,77 · precisão de
área = 0,96; recall = 0,85; F1 = 0,90.

## Tecnologias utilizadas

Python · TensorFlow/Keras · segmentation-models · rasterio · GeoPandas · OpenCV ·
Shapely · scikit-learn · ortoimagens DGT *ortoSat2023* (WMS aberto,
Direção-Geral do Território)

## Sobre os dados e o modelo

As imagens utilizadas pertencem ao serviço aberto de ortoimagens de alta
resolução **ortoSat2023**, da Direção-Geral do Território (DGT), usado com a
devida atribuição. Os polígonos de verdade de terreno, os pesos dos modelos
treinados e os resultados das deteções são propriedade institucional da
CCDR-Norte, I.P. e não são publicados neste repositório. O código é partilhado
como implementação de referência funcional do pipeline completo de treino e
deteção.

---

**Luís Filipe Pacheco** · Engenheiro Sénior e Cientista de Dados,
CCDR-Norte, I.P. · [Perfil no GitHub](https://github.com/LFilipePacheco) ·
[LinkedIn](https://www.linkedin.com/in/lu%C3%ADs-filipe-pacheco-471495b/) ·
[ORCID](https://orcid.org/0009-0001-7676-6542)
