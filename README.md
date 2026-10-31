BR-Risk-ML
Classificação da gravidade de acidentes rodoviários registrados nas rodovias federais brasileiras usando Machine Learning.

O BR-Risk-ML utiliza registros históricos de acidentes disponibilizados pela Polícia Rodoviária Federal (PRF) para classificar a gravidade dos acidentes em três categorias:
- Com Vítimas Fatais
- Com Vítimas Feridas
- Sem Vítimas
O modelo principal utiliza XGBoost e foi desenvolvido em Python, com etapas organizadas de preparação de dados, treinamento, avaliação e inferência.
Escopo: o projeto classifica registros de acidentes já ocorridos. Ele não prevê quando ou onde um acidente acontecerá e não estima o risco de acidentes em locais sem registros.

Objetivos
- Investigar padrões nos registros históricos de acidentes rodoviários.
- Construir um modelo de classificação multiclasse.
- Avaliar o desempenho considerando o desbalanceamento entre classes.
- Organizar um fluxo reproduzível de treinamento e inferência.
- Documentar limitações e decisões metodológicas.
Modelo
O modelo principal é um classificador XGBoost com sete variáveis de entrada:
Tipo	Variáveis
Categóricas	tipo_acidente, causa_acidente, fase_dia
Numéricas	pessoas, veiculos, latitude, longitude


A configuração do modelo está centralizada em [`configs/model_config.json`](configs/model_config.json).
Parâmetros do XGBoost
Parâmetro	Valor
max_depth	8
learning_rate	0.1
n_estimators	400
min_child_weight	3
subsample	0.8
colsample_bytree	0.8
reg_lambda	1


O treinamento utiliza ponderação de amostras para lidar com o desbalanceamento das classes. Também são aplicados multiplicadores por classe, registrados na configuração do modelo.
Dados e divisão temporal
O projeto utiliza registros da PRF referentes ao período de 2017 a 2025.

A divisão temporal adotada no desenvolvimento foi:
Conjunto	Período	Registros
Treinamento	2017–2022	419.209
Validação	2023	67.765
Teste histórico	2024–2025	145.729


No treinamento do artefato atual, treinamento e validação foram combinados, totalizando 486.974 registros de 2017 a 2023.

Atenção metodológica: os dados de 2023 participaram da seleção e otimização do modelo. Além disso, o conjunto de 2024–2025 já foi utilizado em experimentos anteriores. Portanto, os resultados históricos desse conjunto não devem ser apresentados como uma avaliação independente e intocada.

Os arquivos brutos e os conjuntos derivados não precisam ser versionados neste repositório. Consulte a fonte oficial da PRF para obter os dados e reproduzir as etapas de preparação. Os scripts de preparação estão em src/br_risk_ml/data/.
Resultados de validação
Na avaliação do modelo atual sobre o conjunto de validação de 2023:
Métrica	Resultado
Accuracy	0,7746
F1 Macro	0,6143
Balanced Accuracy	0,6199


Métricas por classe
Classe	Precision	Recall	F1-score
Com Vítimas Fatais	0,4445	0,5138	0,4767
Com Vítimas Feridas	0,8570	0,8598	0,8584
Sem Vítimas	0,5312	0,4862	0,5077


O F1 Macro calcula a média do F1-score entre as classes, atribuindo o mesmo peso a cada uma. Isso ajuda a evitar que o desempenho na classe majoritária esconda dificuldades nas classes menos frequentes.

O modelo apresenta melhor desempenho na classe Com Vítimas Feridas. A identificação das classes Com Vítimas Fatais e Sem Vítimas permanece mais desafiadora.

Essas métricas são resultados de desenvolvimento sobre a validação de 2023, que participou da seleção e otimização. Não representam uma estimativa independente de generalização.
Estrutura do projeto
br-risk-ml/
├── archive/                 # Scripts históricos arquivados
├── configs/
│   └── model_config.json    # Configuração do modelo
├── experiments/             # Experimentos e análises
├── reports/
│   └── avaliacao/           # Relatórios de avaliação
├── src/
│   └── br_risk_ml/
│       ├── data/             # Preparação e divisão dos dados
│       ├── models/
│       │   ├── pipeline.py   # Pré-processamento
│       │   ├── train.py      # Treinamento
│       │   ├── evaluate.py   # Avaliação
│       │   └── predict.py    # Inferência
│       └── utils/
│           └── paths.py      # Caminhos do projeto
├── pyproject.toml
├── requirements.txt
└── README.md
Os diretórios data/, artifacts/models/ e resultados experimentais locais podem conter arquivos grandes ou gerados durante a execução. Revise o .gitignore antes de publicar.
Instalação
Requisitos:
- Python 3.10 ou superior.
- pip.
Clone o repositório:
git clone https://github.com/SEU-USUARIO/br-risk-ml.git
cd br-risk-ml
Crie e ative um ambiente virtual.

Windows (PowerShell):
py -m venv .venv
.venv\Scripts\Activate.ps1
Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate
Instale as dependências e o pacote:
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Executar o modelo
Os comandos abaixo pressupõem que os conjuntos preparados estejam disponíveis em data/processed/modelagem/ e que a estrutura de diretórios esperada pelo projeto tenha sido criada.
Treinamento
O treinamento utiliza treino_expandido.csv e validacao_expandido.csv, combinando os registros de 2017 a 2023:
python -m br_risk_ml.models.train
O artefato é salvo em:
artifacts/models/br_risk_xgboost.joblib
Avaliação
A avaliação utiliza validacao_expandido.csv:
python -m br_risk_ml.models.evaluate
O relatório é salvo em:
reports/avaliacao/metricas_validacao.json
Inferência
Para executar o exemplo local incluído no módulo:
python -m br_risk_ml.models.predict
A função prever() recebe um pandas.DataFrame com as sete variáveis utilizadas pelo modelo e retorna a classe prevista e os scores ajustados por classe.

Nota sobre os scores: os valores são obtidos após a aplicação dos multiplicadores por classe e sua normalização para somarem 1 por registro. Eles não devem ser interpretados como probabilidades calibradas.
Limitações e cuidados de interpretação
1. Classificação retrospectiva: o modelo classifica acidentes que já constam na base; não prevê a ocorrência de acidentes.
2. Disponibilidade das variáveis: informações como tipo e causa do acidente, quantidade de pessoas e veículos podem não estar disponíveis antes ou no momento inicial de um evento. Um uso operacional exigiria avaliar quais variáveis estariam disponíveis no instante da previsão.
3. Associação não é causalidade: importância de variáveis e padrões aprendidos não demonstram relações causais.
4. Validação temporal: os resultados de 2023 refletem um processo de desenvolvimento no qual esse período participou da seleção e otimização.
5. Teste histórico: o período 2024–2025 já foi consultado em experimentos anteriores e não deve ser tratado como teste independente.
6. Uso responsável: os resultados não devem ser utilizados isoladamente para decisões operacionais ou políticas públicas sem validação adicional, análise de vieses e avaliação por especialistas.
Tecnologias
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
Licença
A licença do projeto ainda deve ser definida antes da publicação. A licença do código não define automaticamente as condições de uso dos dados de terceiros.
Autor
Projeto de estudo e pesquisa aplicada em Machine Learning, com foco na análise de dados de acidentes rodoviários brasileiros.