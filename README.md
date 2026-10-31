# BR-Risk-ML

Classificação da gravidade de acidentes rodoviários registrados nas rodovias federais brasileiras utilizando Machine Learning e dados públicos da Polícia Rodoviária Federal (PRF).

O BR-Risk-ML utiliza registros históricos de acidentes para classificar cada ocorrência em uma das categorias: **Com Vítimas Fatais**, **Com Vítimas Feridas** ou **Sem Vítimas**.

O modelo principal é baseado em **XGBoost** e foi desenvolvido em Python, com etapas organizadas de consolidação, preparação dos dados, criação dos conjuntos de modelagem, treinamento, avaliação e inferência.

> **Escopo:** o BR-Risk-ML classifica registros de acidentes que já ocorreram. Ele não prevê quando ou onde um acidente acontecerá e não estima o risco de acidentes em locais ou períodos sem registros.

## Objetivos

- Investigar padrões em registros históricos de acidentes rodoviários.
- Desenvolver um modelo de classificação multiclasse.
- Avaliar o desempenho considerando o desbalanceamento entre classes.
- Organizar um fluxo de treinamento e inferência reproduzível.
- Documentar decisões metodológicas, resultados e limitações.

## Modelo

A versão atual utiliza um classificador XGBoost com sete variáveis de entrada:

| Tipo | Variáveis |
|---|---|
| Categóricas | `tipo_acidente`, `causa_acidente`, `fase_dia` |
| Numéricas | `pessoas`, `veiculos`, `latitude`, `longitude` |

A configuração está centralizada em [`configs/model_config.json`](configs/model_config.json).

### Parâmetros do XGBoost

| Parâmetro | Valor |
|---|---:|
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `n_estimators` | 400 |
| `min_child_weight` | 3 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1 |

O treinamento utiliza ponderação de amostras para lidar com o desbalanceamento das classes. Também são aplicados multiplicadores por classe na etapa de decisão, conforme definido em `model_config.json`. O pré-processamento faz parte do pipeline salvo junto ao modelo.

## Dados

São utilizados registros de acidentes da PRF referentes ao período de 2017 a 2025, na modalidade **agrupados por ocorrência**.

- [Portal de Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)
- [Dicionário de dados de acidentes](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dicionario-acidentes)

Os dados brutos e os datasets derivados não são versionados neste repositório. Obtenha os arquivos anuais diretamente no portal oficial da PRF e coloque-os no diretório de dados brutos esperado pelo projeto.

### Divisão temporal utilizada no desenvolvimento

| Conjunto | Período | Registros |
|---|---:|---:|
| Treino inicial | 2017–2022 | 419.209 |
| Validação | 2023 | 67.765 |
| Teste histórico | 2024–2025 | 145.729 |

No treinamento do artefato atual, os conjuntos de treino e validação foram combinados, totalizando **486.974 registros de 2017 a 2023**.

### Ressalvas metodológicas

- O ano de 2023 participou da seleção de variáveis e da otimização. Além disso, o artefato final foi treinado com os registros de 2023. Portanto, as métricas calculadas nesse conjunto são **in-sample**, não uma validação independente.
- O artefato atual não foi treinado com os registros de 2024–2025. Entretanto, esse período já foi consultado em experimentos anteriores. Assim, os resultados são uma avaliação temporal diagnóstica da configuração atual, mas **não representam um holdout independente e intocado**.

## Resultados

O **F1 Macro** é a métrica principal por atribuir o mesmo peso ao F1-score de cada classe. Também são apresentadas Accuracy, Balanced Accuracy e métricas individuais por classe. A acurácia isolada pode ocultar dificuldades nas classes menos frequentes.

### Avaliação temporal diagnóstica — 2024–2025

O artefato atual foi avaliado em 145.729 registros de 2024 e 2025, sem novo treinamento.

| Métrica | Resultado |
|---|---:|
| Accuracy | **0.7567** |
| F1 Macro | **0.5726** |
| Balanced Accuracy | **0.5753** |

#### Métricas por classe

| Classe | Precision | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Com Vítimas Fatais | 0.3840 | 0.4321 | 0.4067 | 10.434 |
| Com Vítimas Feridas | 0.8458 | 0.8507 | 0.8482 | 112.375 |
| Sem Vítimas | 0.4847 | 0.4432 | 0.4630 | 22.920 |

#### Matriz de confusão

Linhas representam as classes reais; colunas representam as classes previstas.

| Classe real / prevista | Com Vítimas Fatais | Com Vítimas Feridas | Sem Vítimas |
|---|---:|---:|---:|
| Com Vítimas Fatais | 4.509 | 5.178 | 747 |
| Com Vítimas Feridas | 6.723 | 95.601 | 10.051 |
| Sem Vítimas | 509 | 12.254 | 10.157 |

O modelo apresenta melhor desempenho na classe **Com Vítimas Feridas**. A identificação de acidentes **Com Vítimas Fatais** e **Sem Vítimas** permanece mais desafiadora. Na classe fatal, o recall de 0.4321 indica que o modelo identificou corretamente cerca de 43% dos registros fatais desse conjunto.

**Interpretação:** como 2024–2025 já foi utilizado em experimentos anteriores, essas métricas servem para documentar e diagnosticar a configuração atual, não para afirmar desempenho em um teste independente.

### Avaliação in-sample — 2023

O valor abaixo foi calculado após o artefato final ter sido treinado com os dados de 2017 a 2023. Portanto, descreve o desempenho sobre dados já utilizados no treinamento e não deve ser comparado diretamente a uma avaliação fora da amostra.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7746 |
| F1 Macro | 0.6143 |
| Balanced Accuracy | 0.6199 |

| Classe | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Com Vítimas Fatais | 0.4445 | 0.5138 | 0.4767 |
| Com Vítimas Feridas | 0.8570 | 0.8598 | 0.8584 |
| Sem Vítimas | 0.5312 | 0.4862 | 0.5077 |

Durante a otimização da ponderação das classes também foi calculada uma estimativa por cross-estimation de **F1 Macro = 0.5643**. Esse valor corresponde a um procedimento diferente da avaliação in-sample de 0.6143; não é uma avaliação independente de todo o processo de desenvolvimento, pois 2023 participou de outras decisões metodológicas.

## Estrutura do projeto

```text
br-risk-ml/
├── configs/
│   └── model_config.json
├── docs/
│   └── experimentos.md
├── experiments/
│   └── ...
├── src/
│   └── br_risk_ml/
│       ├── data/
│       │   ├── consolidar_dados.py
│       │   ├── preparar_dados.py
│       │   ├── criar_datasets.py
│       │   └── dividir_temporal.py
│       ├── models/
│       │   ├── pipeline.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   └── validar_pipeline.py
│       └── utils/
│           └── paths.py
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

Arquivos de dados, modelos treinados e resultados gerados localmente são excluídos do Git por configuração do `.gitignore`. O artefato `.joblib` da versão publicada será disponibilizado separadamente no Hugging Face.

## Instalação

### Requisitos

- Python 3.10 ou superior.
- `pip`.

Clone o repositório:

```bash
git clone https://github.com/llype-ENG/br-risk-ml.git
cd br-risk-ml
```

Crie e ative um ambiente virtual.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências e o pacote:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Reproduzir a preparação dos dados

Com os arquivos anuais da PRF disponíveis no diretório de dados brutos esperado pelo projeto, execute as etapas na seguinte ordem:

```bash
python -m br_risk_ml.data.consolidar_dados
python -m br_risk_ml.data.preparar_dados
python -m br_risk_ml.data.criar_datasets
python -m br_risk_ml.data.dividir_temporal
```

Ao final, o fluxo de modelagem espera encontrar estes arquivos em `data/processed/modelagem/`:

```text
treino_expandido.csv
validacao_expandido.csv
teste_expandido.csv
```

Os nomes e caminhos seguem a implementação dos módulos em `src/br_risk_ml/data/`.

## Treinamento

O treinamento combina `treino_expandido.csv` e `validacao_expandido.csv`, utilizando os registros de 2017 a 2023:

```bash
python -m br_risk_ml.models.train
```

O artefato é salvo em:

```text
artifacts/models/br_risk_xgboost.joblib
```

## Avaliação

O módulo calcula Accuracy, F1 Macro, Balanced Accuracy, métricas por classe e matriz de confusão.

Na versão atual do script, a execução padrão avalia `teste_expandido.csv`:

```bash
python -m br_risk_ml.models.evaluate
```

O relatório de 2024–2025 é salvo em:

```text
reports/avaliacao/metricas_teste_2024_2025.json
```

Consulte as ressalvas metodológicas na seção [Resultados](#resultados) antes de interpretar essas métricas.

## Inferência

Após obter o artefato confiável da versão correspondente (treinando localmente ou baixando-o do repositório de modelos no Hugging Face, quando publicado e vinculado nesta documentação), execute o exemplo incluído no módulo:

```bash
python -m br_risk_ml.models.predict
```

A função `prever()` recebe um `pandas.DataFrame` com as sete variáveis utilizadas pelo modelo e retorna a classe prevista e os scores ajustados por classe.

**Nota sobre os scores:** os valores são obtidos após a aplicação dos multiplicadores por classe e sua normalização para somarem 1 por registro. Eles não devem ser interpretados como probabilidades calibradas.

**Segurança:** arquivos `.joblib` utilizam desserialização Python. Carregue somente artefatos de fontes confiáveis e compatíveis com as versões das bibliotecas usadas no treinamento.

## Análises complementares e evolução

A primeira versão documenta a configuração do modelo e seus resultados de avaliação. Análises adicionais — incluindo comparação com baseline, importância das variáveis do XGBoost final e diagnóstico detalhado dos erros — serão incorporadas à documentação conforme forem concluídas.

O histórico das decisões e dos experimentos relevantes será mantido em `docs/experimentos.md`.

## Limitações e uso responsável

1. **Classificação retrospectiva:** o modelo classifica acidentes já registrados; não prevê a ocorrência de acidentes.
2. **Disponibilidade das variáveis:** informações como tipo e causa do acidente, quantidade de pessoas e veículos podem depender do registro ou da investigação. Um uso operacional exigiria avaliar quais variáveis estariam disponíveis no momento da classificação.
3. **Associação não é causalidade:** a importância de variáveis e os padrões aprendidos não demonstram relações causais.
4. **Desbalanceamento:** o desempenho é significativamente diferente entre as classes, especialmente para acidentes fatais e sem vítimas.
5. **Avaliação temporal:** 2023 foi utilizado no treinamento final e na seleção/otimização. O período de 2024–2025 não foi usado no treinamento do artefato atual, mas já foi consultado em experimentos anteriores.
6. **Ausência de dados de exposição:** a base contém registros de acidentes, não observações completas de locais e períodos sem acidentes ou denominadores de exposição ao tráfego. Portanto, não permite estimar diretamente a probabilidade de ocorrência de acidentes.
7. **Uso responsável:** os resultados não devem ser utilizados isoladamente para decisões operacionais ou políticas públicas sem validação adicional, análise de vieses e avaliação por especialistas.

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- pytest

## Licença

O código-fonte deste projeto é disponibilizado sob a licença [MIT](LICENSE).

A licença MIT se aplica ao código e à documentação de autoria do projeto. Ela não substitui nem altera os termos aplicáveis a dados, bibliotecas ou outros materiais de terceiros. Os dados de acidentes são obtidos do [Portal de Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf).

## Autor

**Fellype Moura** — [GitHub: @llype-ENG](https://github.com/llype-ENG)

Projeto independente de estudo e pesquisa aplicada em Machine Learning, com foco na análise de dados de acidentes rodoviários brasileiros.
