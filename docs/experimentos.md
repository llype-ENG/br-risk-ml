# Histórico de experimentos — BR-Risk-ML

Este documento registra os principais experimentos, decisões metodológicas e diagnósticos do BR-Risk-ML. O objetivo é preservar a rastreabilidade da versão inicial e orientar futuras iterações.

> **Nota metodológica:** os resultados de 2024–2025 são diagnósticos, e não uma avaliação independente, porque esse período já foi consultado durante experimentos anteriores. Não usar esses resultados para alegar generalização independente.

## 1. Dados e divisão temporal

- Período de dados: 2017–2025.
- Registros consolidados: 632.713.
- Registros com variável-alvo: 632.703.
- Registros sem alvo removidos: 10.

| Partição inicial | Período | Registros |
|---|---|---:|
| Treino | 2017–2022 | 419.209 |
| Validação | 2023 | 67.765 |
| Teste histórico | 2024–2025 | 145.729 |

O artefato atual foi treinado com treino + validação, 2017–2023, totalizando 486.974 registros.

**Consequências para avaliação:**
- 2023 participou de decisões de seleção/otimização e do treinamento final; sua avaliação é in-sample.
- 2024–2025 não participou do treinamento do artefato atual, mas foi consultado em experimentos anteriores; sua avaliação é temporal diagnóstica, não holdout intocado.

## 2. Variáveis e prevenção de vazamento

Variáveis utilizadas:
- Categóricas: `tipo_acidente`, `causa_acidente`, `fase_dia`.
- Numéricas: `pessoas`, `veiculos`, `latitude`, `longitude`.

Foram excluídas colunas que revelam diretamente a classificação ou componentes da variável-alvo:
`classificacao_acidente`, `mortos`, `feridos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`.

Observação: embora não sejam colunas diretamente derivadas do alvo, algumas entradas podem refletir informações registradas depois da ocorrência ou durante sua investigação. O modelo é, portanto, retrospectivo.

## 3. Modelo final da versão inicial

Algoritmo: XGBoost multiclasse.

| Parâmetro | Valor |
|---|---:|
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `n_estimators` | 400 |
| `min_child_weight` | 3 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1 |
| `random_state` | 42 |

Balanceamento: `compute_sample_weight("balanced", y) ** 0.5`.

Multiplicadores posteriores de score:
- Fatais: 0.978
- Feridas: 1.000
- Sem vítimas: 1.118

O pipeline inclui imputação numérica pela mediana, preenchimento de categoria ausente com `"Não informado"`, OneHotEncoder (`handle_unknown="ignore"`, `min_frequency=10`) e XGBoost.

## 4. Baseline

Foi avaliada uma baseline trivial:
`DummyClassifier(strategy="most_frequent")`.

Ela sempre prevê a classe majoritária, `Com Vítimas Feridas`.

Resultados no conjunto diagnóstico 2024–2025:

| Métrica | Baseline |
|---|---:|
| Accuracy | 0.7711 |
| F1 Macro | 0.2903 |
| Balanced Accuracy | 0.3333 |

O F1 por classe da baseline foi:
- Com Vítimas Fatais: 0.0000
- Com Vítimas Feridas: 0.8708
- Sem Vítimas: 0.0000

A acurácia relativamente alta decorre da prevalência da classe feridos. A baseline não reconhece as classes minoritárias.

## 5. Avaliação do XGBoost

Avaliação temporal diagnóstica no período 2024–2025:

| Métrica | XGBoost |
|---|---:|
| Accuracy | 0.7567 |
| F1 Macro | 0.5726 |
| Balanced Accuracy | 0.5753 |

Comparação:

| Métrica | Baseline | XGBoost |
|---|---:|---:|
| Accuracy | 0.7711 | 0.7567 |
| F1 Macro | 0.2903 | 0.5726 |
| Balanced Accuracy | 0.3333 | 0.5753 |

Variação do F1 Macro: +0.2823 ponto absoluto. O XGBoost tem acurácia 0.0144 menor, mas apresenta melhor equilíbrio entre as classes.

### Métricas por classe

| Classe | Precision | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Fatais | 0.3840 | 0.4321 | 0.4067 | 10.434 |
| Feridas | 0.8458 | 0.8507 | 0.8482 | 112.375 |
| Sem vítimas | 0.4847 | 0.4432 | 0.4630 | 22.920 |

### Matriz de confusão

Linhas = classes reais; colunas = classes previstas.

| Real / Prevista | Fatais | Feridas | Sem vítimas |
|---|---:|---:|---:|
| Fatais | 4.509 | 5.178 | 747 |
| Feridas | 6.723 | 95.601 | 10.051 |
| Sem vítimas | 509 | 12.254 | 10.157 |

### Avaliação de 2023

Após treinar o artefato final com 2017–2023, a avaliação sobre 2023 apresentou:
- Accuracy: 0.7746
- F1 Macro: 0.6143
- Balanced Accuracy: 0.6199

Esses valores são in-sample. Também foi obtida estimativa por cross-estimation de F1 Macro = 0.564327 durante a otimização da ponderação das classes. Como 2023 entrou em outras decisões de desenvolvimento, esse valor também não constitui estimativa independente de todo o processo.

## 6. Importância de variáveis — Gain

A importância nativa do XGBoost foi calculada sobre as colunas transformadas e agregada novamente às sete variáveis originais.

O pré-processamento gerou 119 colunas one-hot/numéricas; 118 foram utilizadas em divisões das árvores.

| Variável original | Gain total | Participação no Gain |
|---|---:|---:|
| `tipo_acidente` | 485.874,6110 | 42,93% |
| `causa_acidente` | 141.144,3859 | 12,47% |
| `pessoas` | 140.384,6276 | 12,40% |
| `latitude` | 130.033,8407 | 11,49% |
| `longitude` | 111.504,1821 | 9,85% |
| `veiculos` | 75.828,1492 | 6,70% |
| `fase_dia` | 47.037,4173 | 4,16% |

Interpretação: `tipo_acidente` domina a participação no Gain. Essa métrica resume a contribuição das divisões das árvores segundo o critério interno do XGBoost; não é causalidade.

## 7. Análise SHAP

A análise foi executada com:
- amostra de 2.000 registros de 2024–2025;
- `random_state=42`;
- TreeExplainer;
- 119 colunas transformadas;
- saída SHAP para três classes;
- agregação das contribuições das colunas transformadas por variável original.

### Importância global

| Variável | Mean absolute SHAP |
|---|---:|
| `causa_acidente` | 2.313312 |
| `tipo_acidente` | 1.208817 |
| `latitude` | 0.434905 |
| `longitude` | 0.391718 |
| `fase_dia` | 0.326366 |
| `veiculos` | 0.251389 |
| `pessoas` | 0.167462 |

A ordenação difere do Gain: SHAP coloca `causa_acidente` em primeiro lugar, enquanto Gain coloca `tipo_acidente` em primeiro lugar. Não há contradição: Gain mede contribuição das divisões; mean absolute SHAP resume a magnitude média das contribuições às saídas do modelo nos exemplos amostrados.

**Limitação:** SHAP explica as saídas internas do XGBoost antes da aplicação dos multiplicadores posteriores de score por classe. Os valores SHAP não devem ser lidos como variação direta na probabilidade final.

Arquivos produzidos pelo experimento:
- `shap_importancia_global.csv`
- `shap_importancia_por_classe.csv`
- `shap_metadata.json`
- `shap_global.png`
- gráficos por classe

## 8. Diagnóstico de erros

O diagnóstico de erros foi executado no conjunto temporal 2024–2025 com o artefato atual e a regra de predição do projeto.

### Classe fatal

Total de fatais reais: 10.434.
- Preditos corretamente como fatais: 4.509.
- Preditos como feridos: 5.178.
- Preditos como sem vítimas: 747.
- Recall fatal: 43,21%.

### Tipos de acidente

| Tipo | Fatais reais | Fatais reconhecidos | Recall fatal |
|---|---:|---:|---:|
| Engavetamento | 67 | 3 | 4,48% |
| Capotamento | 116 | 6 | 5,17% |
| Colisão lateral mesmo sentido | 414 | 27 | 6,52% |
| Saída de leito carroçável | 1.238 | 83 | 6,70% |
| Colisão transversal | 852 | 63 | 7,39% |
| Atropelamento de animal | 143 | 13 | 9,09% |
| Colisão com objeto | 557 | 66 | 11,85% |
| Colisão traseira | 1.184 | 148 | 12,50% |
| Tombamento | 531 | 67 | 12,62% |
| Colisão frontal | 2.862 | 2.433 | 85,01% |
| Atropelamento de pedestre | 1.848 | 1.501 | 81,22% |

Colisão traseira, saída de leito carroçável e colisão transversal somaram 2.622 falsos negativos fatais (50,64% dos 5.178 classificados como feridos).

É importante separar:
- **volume de erros:** quantos erros uma categoria acumula;
- **taxa de erro/recall:** qual fração dos fatais naquela categoria foi reconhecida.

Uma categoria pequena pode ter recall muito baixo sem representar grande volume absoluto de erros.

### Causas do acidente

Exemplos de categorias com baixo recall fatal:
- Condutor deixou de manter distância do veículo à frente: 157 fatais, 3 reconhecidos (1,91%).
- Falhas mecânicas/elétricas: 105 fatais, 4 reconhecidos (3,81%).
- Deixar de dar preferência em interseção: 47 fatais, 2 reconhecidos (4,26%).
- Reação tardia ou ineficiente: 1.115 fatais, 203 reconhecidos (18,21%).
- Ausência de reação do condutor: 1.468 fatais, 354 reconhecidos (24,11%).

Entre as causas com mais falsos negativos em volume, destacaram-se ausência de reação (945), reação tardia ou ineficiente (819) e acesso à via sem observar outros veículos (619).

### Fase do dia

| Fase | Fatais reais | Reconhecidos | Recall fatal |
|---|---:|---:|---:|
| Pleno dia | 4.046 | 1.290 | 31,88% |
| Anoitecer | 540 | 211 | 39,07% |
| Amanhecer | 760 | 376 | 49,47% |
| Plena noite | 5.088 | 2.632 | 51,73% |

A diferença é descritiva e pode refletir a composição por tipo, causa, localização ou outras características. Não permite atribuir causalidade à fase do dia.

### Combinações de causa e tipo

Exemplos:
- Acessar a via sem observar outros veículos + colisão transversal: 457 fatais, 22 reconhecidos (4,81%).
- Reação tardia/ineficiente + saída de leito carroçável: 246 fatais, 10 reconhecidos (4,07%).
- Ausência de reação + saída de leito carroçável: 341 fatais, 35 reconhecidos (10,26%).
- Condutor deixou de manter distância + colisão traseira: 118 fatais, nenhum reconhecido (0%).
- Transitar na contramão + colisão frontal: 1.234 fatais, 1.159 reconhecidos (93,92%).

Esses cruzamentos apontam subgrupos em que a classe fatal foi mais difícil ou mais fácil de reconhecer. Não demonstram causalidade.

### UF

Foram observadas diferenças de composição e de recall entre UFs, mas amostras por estado variam e a distribuição dos tipos e causas também pode variar. Não se deve interpretar esses resultados como evidência de risco real por estado ou como comparação definitiva de qualidade dos registros estaduais.

## 9. Decisões e próximos experimentos

Para a versão inicial:
- manter as sete variáveis definidas;
- manter a configuração atual de XGBoost;
- preservar o artefato atual;
- documentar os resultados e limitações;
- não realizar novos ajustes com base no conjunto de diagnóstico 2024–2025.

Para versões futuras:
1. criar uma nova avaliação temporal realmente independente, com período não consultado durante desenvolvimento;
2. investigar estratégias de balanceamento usando somente dados de desenvolvimento;
3. avaliar outras famílias de algoritmos sob protocolo temporal consistente;
4. testar novas variáveis somente quando sua disponibilidade no momento de uso estiver definida;
5. verificar calibração e utilidade operacional antes de qualquer aplicação de alto impacto.

## 10. Artefatos do modelo

Arquivo principal:
`artifacts/models/br_risk_xgboost.joblib`

O objeto serializado contém o pipeline, a lista de classes, as variáveis, configurações do modelo e metadados de balanceamento e pré-processamento.

O artefato deve ser carregado apenas de fonte confiável, pois arquivos `joblib` utilizam desserialização Python.
