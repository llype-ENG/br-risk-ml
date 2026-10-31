
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERICAS = [
    "km",
    "hora",
    "mes",
    "latitude",
    "longitude",
]

CATEGORICAS = [
    "uf",
    "br",
    "municipio",
    "dia_semana",
    "condicao_metereologica",
    "tipo_pista",
    "uso_solo",
    "fase_dia",
    "sentido_via",
]

CATEGORICAS_EXPANDIDO = CATEGORICAS + [
    "tipo_acidente",
    "causa_acidente",
]

NUMERICAS_EXPANDIDO = NUMERICAS + [
    "pessoas",
    "veiculos",
]


def criar_preprocessador(
    numericas=None,
    categoricas=None,
):
    if numericas is None:
        numericas = NUMERICAS

    if categoricas is None:
        categoricas = CATEGORICAS

    pipeline_numerico = Pipeline(
        steps=[
            (
                "imputacao",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            (
                "imputacao",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Não informado",
                ),
            ),
            (
                "codificacao",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, numericas),
            ("categoricas", pipeline_categorico, categoricas),
        ]
    )

    return preprocessador


def obter_variaveis(nome):
    if nome == "basico":
        return NUMERICAS, CATEGORICAS

    if nome == "expandido":
        return NUMERICAS_EXPANDIDO, CATEGORICAS_EXPANDIDO

    raise ValueError(
        "Nome inválido. Utilize 'basico' ou 'expandido'."
    )
