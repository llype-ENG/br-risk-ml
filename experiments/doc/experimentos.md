# Histórico de Experimentos --- BR-Risk-ML

Este documento registra as principais etapas de investigação e decisões
técnicas que levaram à configuração atual do BR-Risk-ML.

O projeto classifica retrospectivamente a gravidade de acidentes
registrados pela Polícia Rodoviária Federal (PRF). Ele não estima a
probabilidade de um acidente ocorrer em determinado local ou momento: a
base contém registros de acidentes, sem observações de exposição ou de
ausência de acidentes.

## 1. Definição do problema

A variável-alvo possui três classes:

-   `Com Vítimas Fatais`
-   `Com Vítimas Feridas`
-   `Sem Vítimas`

A distribuição das classes é desigual, com predominância de acidentes
com vítimas feridas. Por isso, o **F1 Macro** foi adotado como métrica
principal, acompanhado de Balanced Accuracy e métricas individuais de
precisão, recall e F1 por classe.

## 2. Preparação e divisão temporal

Foram utilizados registros da PRF de 2017 a 2025. O conjunto consolidado
possui 632.713 registros; após remover 10 registros sem variável-alvo,
restaram 632.703.

  Partição               Período   Registros
  ----------------- ------------ -----------
  Treino inicial      2017--2022     419.209
  Validação                 2023      67.765
  Teste histórico     2024--2025     145.729

A separação temporal aproxima a avaliação de um cenário em que dados
históricos são usados para classificar registros de períodos
posteriores.

**Ressalva:** os dados de 2024--2025 já foram utilizados em experimentos
anteriores de comparação entre modelos. Portanto, não devem ser
apresentados como um teste independente e intocado. O ano de 2023 também
participou da seleção de variáveis e otimização, o que pode tornar os
resultados de validação otimistas.

## 3. Seleção de variáveis

Foram realizadas análises de importância e experimentos com diferentes
conjuntos de atributos. A configuração selecionada utiliza sete
variáveis:

  Variável           Tipo
  ------------------ ------------
  `tipo_acidente`    Categórica
  `causa_acidente`   Categórica
  `fase_dia`         Categórica
  `pessoas`          Numérica
  `veiculos`         Numérica
  `latitude`         Numérica
  `longitude`        Numérica

Para evitar vazamento de informação, foram excluídas variáveis
diretamente relacionadas ao resultado do acidente:

-   `classificacao_acidente`
-   `mortos`
-   `feridos`
-   `feridos_leves`
-   `feridos_graves`
-   `ilesos`
-   `ignorados`

A importância das variáveis representa associação preditiva observada
pelo modelo, não causalidade.

## 4. Comparação de algoritmos

Foram avaliadas abordagens com Random Forest e XGBoost, além de
diferentes conjuntos de atributos e estratégias de balanceamento.

Em experimentos históricos realizados sobre o conjunto de 2024--2025,
foram observados:

  Experimento                           F1 Macro histórico
  ----------------------------------- --------------------
  Random Forest                                     0,5077
  XGBoost com ponderação balanceada                 0,5091

Esses resultados pertencem a configurações anteriores e não representam
necessariamente o desempenho da configuração atual. Como 2024--2025 foi
consultado durante os experimentos, esses valores são referências
históricas, não uma estimativa independente de generalização.

O XGBoost foi escolhido para a configuração atual por oferecer uma
abordagem adequada aos dados tabulares e permitir investigar estratégias
de ponderação das classes.

## 5. Balanceamento e otimização

Devido ao desbalanceamento entre as classes, foram testados pesos
derivados de `compute_sample_weight("balanced", y)`, elevados aos
expoentes:

-   0,00
-   0,25
-   0,50
-   0,75
-   1,00

Também foram investigados multiplicadores aplicados aos scores de
probabilidade após o treinamento, com foco nas classes
`Com Vítimas Fatais` e `Sem Vítimas`.

A otimização foi conduzida usando a validação de 2023. A estimativa
denominada "honesta" utilizou uma divisão interna para estimar os
multiplicadores, reduzindo a dependência de uma única avaliação.

A configuração selecionada utiliza expoente de ponderação igual a 0,5:

  Classe                    Multiplicador
  ----------------------- ---------------
  `Com Vítimas Fatais`              0,978
  `Com Vítimas Feridas`             1,000
  `Sem Vítimas`                     1,118

Esses multiplicadores ajustam os scores usados na decisão final; não
alteram os parâmetros internos já aprendidos pelo XGBoost.

## 6. Configuração atual do XGBoost

A configuração oficial está registrada em `configs/model_config.json`:

  Parâmetro              Valor
  -------------------- -------
  `max_depth`                8
  `learning_rate`          0,1
  `n_estimators`           400
  `min_child_weight`         3
  `subsample`              0,8
  `colsample_bytree`       0,8
  `reg_lambda`               1
  `random_state`            42

O treinamento final combina os dados de 2017 a 2023, totalizando 486.974
registros. O pré-processamento inclui imputação de valores numéricos e
codificação One-Hot das variáveis categóricas.

Este documento descreve o histórico e as motivações dos experimentos. A
configuração executável deve ser consultada em
`configs/model_config.json`.

## 7. Avaliação na validação de 2023

Na avaliação completa da configuração final sobre 2023, foram obtidos:

  Métrica               Resultado
  ------------------- -----------
  Acurácia                 0,7746
  F1 Macro                 0,6143
  Balanced Accuracy        0,6199

Métricas por classe:

  Classe                  Precisão   Recall       F1
  --------------------- ---------- -------- --------
  Com Vítimas Fatais        0,4445   0,5138   0,4767
  Com Vítimas Feridas       0,8570   0,8598   0,8584
  Sem Vítimas               0,5312   0,4862   0,5077

A diferença entre o desempenho da classe majoritária e o das demais
evidencia espaço para melhorar a identificação de acidentes fatais e sem
vítimas.

O arquivo `configs/model_config.json` também registra uma estimativa
honesta de F1 Macro igual a 0,5643. Esse valor vem de um procedimento
diferente da avaliação completa de 2023 e não deve ser confundido com o
F1 Macro de 0,6143.

## 8. Limitações e cuidados de interpretação

-   **Validação reutilizada:** 2023 participou da seleção de variáveis e
    otimização. Seus resultados não equivalem a uma avaliação totalmente
    independente.
-   **Teste histórico reutilizado:** 2024--2025 já foi consultado em
    experimentos anteriores. Não deve ser divulgado como holdout
    independente.
-   **Disponibilidade das variáveis:** `tipo_acidente`,
    `causa_acidente`, `pessoas` e `veiculos` podem depender do registro
    ou da investigação do acidente. A configuração atual deve ser
    interpretada como classificação retrospectiva, não como previsão
    necessariamente disponível antes ou no instante da ocorrência.
-   **Associação não é causalidade:** importância de variável não prova
    que ela causa maior ou menor gravidade.
-   **Incerteza:** intervalos calculados por bootstrap em linhas tratam
    registros como observações independentes e podem não representar
    toda a dependência temporal ou geográfica.
-   **Escopo dos dados:** a base não contém denominadores de exposição
    ou observações de locais e períodos sem acidentes. Portanto, não
    permite estimar diretamente o risco de ocorrência de acidentes.

## 9. Reprodutibilidade

A configuração oficial está em:

``` text
configs/model_config.json
```

O fluxo de treinamento e inferência mantido pelo projeto está em:

``` text
src/br_risk_ml/
```

Os scripts exploratórios não fazem parte do fluxo oficial de execução.
Este documento preserva as principais decisões e resultados para dar
transparência ao desenvolvimento sem exigir que o usuário execute todos
os experimentos históricos.

------------------------------------------------------------------------

*Documento de apoio à documentação técnica do BR-Risk-ML. Consulte o
README principal para instalação, estrutura do projeto e instruções de
uso.*
