# BR-Risk-ML

Classificação da gravidade de acidentes rodoviários registrados nas rodovias federais brasileiras utilizando Machine Learning e dados públicos da Polícia Rodoviária Federal (PRF).

O BR-Risk-ML utiliza registros históricos de acidentes para classificar cada ocorrência em uma das categorias:

- **Com Vítimas Fatais**
- **Com Vítimas Feridas**
- **Sem Vítimas**

O modelo principal é um classificador XGBoost desenvolvido em Python, com etapas de consolidação, preparação dos dados, treinamento, avaliação, análise de erros e inferência.

> **Escopo:** o BR-Risk-ML classifica registros de acidentes que já ocorreram. Ele não prevê quando ou onde um acidente acontecerá e não estima o risco de acidentes em locais ou períodos sem registros.

**Modelo no Hugging Face:** [fellype-dev-ai/br-risk-ml](https://huggingface.co/fellype-dev-ai/br-risk-ml)  
**Código-fonte:** [llype-ENG/br-risk-ml](https://github.com/llype-ENG/br-risk-ml)

---

## Objetivos

- Investigar padrões em registros históricos de acidentes rodoviários.
- Desenvolver um modelo de classificação multiclasse.
- Avaliar o desempenho considerando o desbalanceamento entre classes.
- Comparar o modelo com uma baseline simples.
- Investigar a contribuição das variáveis e os padrões associados aos erros.
- Organizar um fluxo de treinamento e inferência reproduzível.
- Documentar decisões metodológicas, resultados e limitações.

## Dados

O projeto utiliza registros de acidentes da PRF referentes ao período de **2017 a 2025**, agrupados por ocorrência.

Fontes:
- [Portal de Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos)
- [Dados abertos de acidentes da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)

Os dados brutos e os conjuntos derivados não são versionados neste repositório. Obtenha os arquivos diretamente nas fontes oficiais e siga os scripts de preparação presentes em `src/br_risk_ml/data/`.

### Consolidação e divisão temporal

Após a consolidação, foram contabilizados **632.713 registros**, dos quais **632.703** possuíam a variável-alvo disponível. Dez registros sem alvo foram removidos da modelagem.

| Conjunto | Período | Registros |
|---|---:|---:|
| Treino inicial | 2017–2022 | 419.209 |
| Validação de desenvolvimento | 2023 | 67.765 |
| Teste histórico | 2024–2025 | 145.729 |

Para treinar o artefato atual, os conjuntos de treino e validação foram combinados, totalizando **486.974 registros de 2017 a 2023**.

### Ressalvas metodológicas sobre os períodos

- O ano de 2023 participou da seleção de variáveis e da otimização. Além disso, o artefato final foi treinado com os registros de 2023. Portanto, a avaliação desse período é **in-sample**, não uma validação independente.
- O artefato atual não foi treinado com os registros de 2024–2025. Entretanto, esse período já foi consultado em experimentos anteriores. Assim, os resultados são uma **avaliação temporal diagnóstica**, não um holdout independente e intocado.
- Não se deve interpretar os resultados apresentados como uma estimativa imparcial de desempenho em produção.

## Variáveis do modelo

O modelo utiliza sete variáveis de entrada:

| Tipo | Variáveis |
|---|---|
| Categóricas | `tipo_acidente`, `causa_acidente`, `fase_dia` |
| Numéricas | `pessoas`, `veiculos`, `latitude`, `longitude` |

Colunas que revelam diretamente a classificação ou seus componentes — como `classificacao_acidente`, `mortos`, `feridos`, `feridos_leves`, `feridos_graves`, `ilesos` e `ignorados` — foram excluídas das entradas para evitar vazamento do alvo.

O pré-processamento faz parte do pipeline salvo junto ao modelo:
- imputação de valores numéricos ausentes pela mediana;
- preenchimento de categorias ausentes com `"Não informado"`;
- codificação one-hot das variáveis categóricas;
- tratamento de categorias desconhecidas e agrupamento de categorias pouco frequentes.

## Modelo

O artefato principal utiliza XGBoost com os seguintes parâmetros:

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

O treinamento utiliza pesos de amostra calculados a partir do balanceamento das classes, aplicando a raiz quadrada desses pesos. Após o treinamento, são aplicados multiplicadores de score por classe:

| Classe | Multiplicador |
|---|---:|
| Com Vítimas Fatais | 0.978 |
| Com Vítimas Feridas | 1.000 |
| Sem Vítimas | 1.118 |

Os multiplicadores fazem parte da regra de decisão da implementação. Os scores finais normalizados não devem ser interpretados como probabilidades calibradas.

## Resultados

O **F1 Macro** é a métrica principal, pois atribui o mesmo peso ao F1-score de cada classe. A acurácia é apresentada em conjunto com Balanced Accuracy e métricas por classe, para não esconder o desempenho das classes minoritárias.

### Comparação com baseline — avaliação diagnóstica 2024–2025

A baseline utiliza `DummyClassifier(strategy="most_frequent")`, que sempre prevê a classe majoritária observada no conjunto de treinamento.

| Métrica | Baseline | XGBoost |
|---|---:|---:|
| Accuracy | 0.7711 | 0.7567 |
| F1 Macro | 0.2903 | **0.5726** |
| Balanced Accuracy | 0.3333 | **0.5753** |

O XGBoost obteve aumento de **0.2823 ponto absoluto em F1 Macro** em relação à baseline. A baseline apresenta acurácia ligeiramente maior porque prevê sempre a classe majoritária, mas não identifica as outras duas classes. O resultado ilustra por que a acurácia isolada não é suficiente para avaliar este problema.

### Avaliação temporal diagnóstica — 2024–2025

O artefato atual foi avaliado em 145.729 registros de 2024 e 2025, sem novo treinamento nesse período.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7567 |
| F1 Macro | **0.5726** |
| Balanced Accuracy | 0.5753 |

#### Métricas por classe

| Classe | Precision | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Com Vítimas Fatais | 0.3840 | 0.4321 | 0.4067 | 10.434 |
| Com Vítimas Feridas | 0.8458 | 0.8507 | 0.8482 | 112.375 |
| Sem Vítimas | 0.4847 | 0.4432 | 0.4630 | 22.920 |

#### Matriz de confusão

Linhas representam as classes reais; colunas representam as classes previstas.

| Classe real / prevista | Fatais | Feridas | Sem vítimas |
|---|---:|---:|---:|
| Com Vítimas Fatais | 4.509 | 5.178 | 747 |
| Com Vítimas Feridas | 6.723 | 95.601 | 10.051 |
| Sem Vítimas | 509 | 12.254 | 10.157 |

O modelo apresenta melhor desempenho na classe **Com Vítimas Feridas**. A identificação das classes **Com Vítimas Fatais** e **Sem Vítimas** permanece mais desafiadora. Na classe fatal, o recall de 0.4321 indica que o modelo identificou corretamente cerca de 43% dos registros fatais avaliados.

> Como 2024–2025 já foi utilizado em experimentos anteriores, essas métricas servem para documentar e diagnosticar a configuração atual. Não devem ser apresentadas como resultado de um teste independente.

### Avaliação in-sample — 2023

O artefato final foi treinado com os dados de 2017 a 2023. Portanto, os resultados abaixo são in-sample:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7746 |
| F1 Macro | 0.6143 |
| Balanced Accuracy | 0.6199 |

Esses valores descrevem o ajuste do modelo aos dados usados no treinamento e não representam uma estimativa independente de generalização.

Durante a otimização da ponderação das classes também foi calculada uma estimativa por cross-estimation de F1 Macro = **0.5643**. Trata-se de um procedimento diferente da avaliação in-sample; não é uma avaliação independente de todo o processo de desenvolvimento, pois 2023 participou de outras decisões metodológicas.

## Análises de interpretabilidade e diagnóstico

As análises abaixo foram realizadas sobre o artefato atual. As análises que utilizaram 2024–2025 têm caráter diagnóstico, pelas ressalvas descritas anteriormente.

### Importância por Gain

A importância nativa do XGBoost foi agregada das colunas transformadas pelo one-hot encoding para as sete variáveis originais.

| Variável | Participação no Gain |
|---|---:|
| `tipo_acidente` | 42.93% |
| `causa_acidente` | 12.47% |
| `pessoas` | 12.40% |
| `latitude` | 11.49% |
| `longitude` | 9.85% |
| `veiculos` | 6.70% |
| `fase_dia` | 4.16% |

O Gain representa a contribuição das divisões das árvores segundo a medida interna do XGBoost. Não representa causalidade nem deve ser interpretado isoladamente como impacto sobre a previsão.

### SHAP

A análise SHAP foi feita com uma amostra de 2.000 registros do período 2024–2025. As contribuições das colunas one-hot foram agrupadas novamente por variável original.

| Variável | Média do valor absoluto de SHAP |
|---|---:|
| `causa_acidente` | 2.3133 |
| `tipo_acidente` | 1.2088 |
| `latitude` | 0.4349 |
| `longitude` | 0.3917 |
| `fase_dia` | 0.3264 |
| `veiculos` | 0.2514 |
| `pessoas` | 0.1675 |

Gain e SHAP medem aspectos diferentes. O Gain resume a contribuição das divisões das árvores; o SHAP estima a magnitude das contribuições das variáveis para as saídas do modelo nos exemplos analisados. Por isso, a ordem pode ser diferente. Os valores SHAP apresentados explicam as saídas internas do XGBoost antes dos multiplicadores de score por classe.

### Diagnóstico dos erros da classe fatal

Na avaliação diagnóstica de 2024–2025, havia 10.434 registros fatais:
- 4.509 foram identificados corretamente como fatais;
- 5.178 foram classificados como acidentes com feridos;
- 747 foram classificados como acidentes sem vítimas.

Entre os 5.178 falsos negativos fatais, três tipos concentraram 50.64% dos erros:

| Tipo de acidente | Fatais classificados como feridos |
|---|---:|
| Colisão traseira | 960 |
| Saída de leito carroçável | 880 |
| Colisão transversal | 782 |

O recall fatal variou bastante entre tipos de acidente. Exemplos:
- Colisão frontal: 85.01% (2.433 reconhecidos em 2.862 fatais).
- Atropelamento de pedestre: 81.22% (1.501 em 1.848).
- Colisão transversal: 7.39% (63 em 852).
- Saída de leito carroçável: 6.70% (83 em 1.238).

Também foram observadas diferenças por fase do dia: recall fatal de 31.88% durante o pleno dia e 51.73% durante a plena noite. Esses resultados indicam diferenças na composição dos erros, mas não demonstram que o horário ou o tipo de acidente causem a classificação incorreta.

O diagnóstico por causa apontou concentrações de erros em categorias como ausência de reação do condutor, reação tardia ou ineficiente e acesso à via sem observar outros veículos. São associações observadas nos registros avaliados, não relações causais.

As análises por tipo, causa, UF e fase do dia estão documentadas em `docs/experimentos.md`.

## Estrutura do projeto

```text
br-risk-ml/
├── archive/                 # Scripts históricos arquivados
├── configs/
│   └── model_config.json    # Configuração do modelo
├── docs/
│   └── experimentos.md      # Histórico e resultados dos experimentos
├── experiments/             # Scripts de análise experimental
├── src/
│   └── br_risk_ml/
│       ├── data/            # Consolidação e preparação dos dados
│       ├── models/
│       │   ├── pipeline.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── predict.py
│       └── utils/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

Os diretórios de dados, os artefatos treinados e relatórios locais podem ser gerados durante a execução e não precisam ser versionados no Git. O artefato publicado é disponibilizado no Hugging Face.

## Instalação

### Requisitos

- Python 3.10 ou superior.
- `pip`.

Clone o repositório:

```powershell
git clone https://github.com/llype-ENG/br-risk-ml.git
cd br-risk-ml
```

Crie e ative um ambiente virtual:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale as dependências e o pacote:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Reproduzir a preparação dos dados

Com os arquivos anuais da PRF disponíveis no diretório bruto esperado pelo projeto, execute:

```powershell
python -m br_risk_ml.data.consolidar_dados
python -m br_risk_ml.data.preparar_dados
python -m br_risk_ml.data.criar_datasets
python -m br_risk_ml.data.dividir_temporal
```

O fluxo de modelagem espera os arquivos preparados em `data/processed/modelagem/`:

```text
treino_expandido.csv
validacao_expandido.csv
teste_expandido.csv
```

## Treinamento e avaliação

Treinamento:

```powershell
python -m br_risk_ml.models.train
```

O artefato é salvo em:

```text
artifacts/models/br_risk_xgboost.joblib
```

Avaliação padrão do código atual:

```powershell
python -m br_risk_ml.models.evaluate
```

Consulte as ressalvas metodológicas antes de interpretar os resultados gerados.

## Inferência e download do artefato

O modelo treinado está disponível em:

[Hugging Face — fellype-dev-ai/br-risk-ml](https://huggingface.co/fellype-dev-ai/br-risk-ml)

É possível baixar o arquivo com `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="fellype-dev-ai/br-risk-ml",
    filename="br_risk_xgboost.joblib",
)

artifact = joblib.load(model_path)
pipeline = artifact["pipeline"]
```

O pipeline acima contém o pré-processamento e o estimador. Para reproduzir a regra de decisão utilizada pelo projeto, incluindo os multiplicadores por classe, consulte a implementação `prever()` em `src/br_risk_ml/models/predict.py`.

**Segurança:** arquivos `.joblib` utilizam desserialização Python. Carregue apenas artefatos de fontes confiáveis e em ambiente com dependências compatíveis.

## Limitações e uso responsável

1. **Classificação retrospectiva:** o modelo classifica acidentes já registrados; não prevê a ocorrência de acidentes.
2. **Disponibilidade das variáveis:** informações como tipo e causa do acidente, quantidade de pessoas e veículos podem depender do registro ou da investigação. Um uso operacional exigiria avaliar quais variáveis estariam disponíveis no momento da classificação.
3. **Associação não é causalidade:** importância de variáveis e padrões aprendidos não demonstram relações causais.
4. **Desbalanceamento:** o desempenho difere significativamente entre as classes, especialmente para acidentes fatais e sem vítimas.
5. **Validação temporal:** 2023 foi utilizado no treinamento final e na seleção/otimização. O período de 2024–2025 não foi usado no treinamento do artefato atual, mas já foi consultado em experimentos anteriores.
6. **Ausência de dados de exposição:** a base contém registros de acidentes, não observações completas de locais e períodos sem acidentes ou denominadores de exposição ao tráfego. Portanto, não permite estimar diretamente a probabilidade de ocorrência de acidentes.
7. **Uso responsável:** os resultados não devem ser utilizados isoladamente para decisões operacionais, fiscalização, seguros ou políticas públicas sem validação adicional, análise de vieses e avaliação por especialistas.

## Tecnologias

Python · Pandas · NumPy · Scikit-learn · XGBoost · Joblib · SHAP · pytest

## Licença

O código-fonte e a documentação de autoria do projeto são disponibilizados sob a licença [MIT](LICENSE).

A licença MIT não substitui nem altera os termos aplicáveis aos dados da PRF, bibliotecas ou outros materiais de terceiros. Consulte as condições das respectivas fontes.

## Autor

**Fellype Moura** — [GitHub: @llype-ENG](https://github.com/llype-ENG)

Projeto independente de estudo e pesquisa aplicada em Machine Learning, com foco na análise de dados de acidentes rodoviários brasileiros.
