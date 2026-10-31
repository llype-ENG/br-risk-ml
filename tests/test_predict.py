
import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from br_risk_ml.models.predict import prever


FEATURES_CATEGORICAS = [
    "tipo_acidente",
    "causa_acidente",
    "fase_dia",
]

FEATURES_NUMERICAS = [
    "pessoas",
    "veiculos",
    "latitude",
    "longitude",
]

FEATURES = FEATURES_CATEGORICAS + FEATURES_NUMERICAS

CLASSES = [
    "Com Vítimas Fatais",
    "Com Vítimas Feridas",
    "Sem Vítimas",
]


class PipelineMock:
    """Simula o pipeline treinado do XGBoost."""

    def predict_proba(self, dados):
        return np.array(
            [[0.10, 0.60, 0.30]] * len(dados)
        )


def criar_entrada():
    """Cria um registro fictício para os testes."""

    return pd.DataFrame([{
        "tipo_acidente": "Colisão traseira",
        "causa_acidente": "Falta de atenção à condução",
        "fase_dia": "Pleno dia",
        "pessoas": 2,
        "veiculos": 2,
        "latitude": -15.7801,
        "longitude": -47.9292,
    }])


def criar_artefato_mock():
    """Reproduz a estrutura do artefato oficial."""

    return {
        "pipeline": PipelineMock(),
        "classes": CLASSES,
        "features": FEATURES,
        "features_categoricas": FEATURES_CATEGORICAS,
        "features_numericas": FEATURES_NUMERICAS,
        "pesos_pos_treino": {
            "Com Vítimas Fatais": 0.978,
            "Sem Vítimas": 1.118,
            "demais": 1.0,
        },
    }


@pytest.fixture
def modelo_mock():
    """Simula o carregamento do modelo."""

    with patch(
        "br_risk_ml.models.predict.carregar_modelo",
        return_value=criar_artefato_mock(),
    ):
        yield


def test_prever_retorna_dataframe(modelo_mock):
    resultado = prever(criar_entrada())

    assert isinstance(resultado, pd.DataFrame)


def test_prever_retorna_classe_valida(modelo_mock):
    resultado = prever(criar_entrada())

    assert resultado.loc[0, "classe_prevista"] in CLASSES


def test_prever_retorna_scores(modelo_mock):
    resultado = prever(criar_entrada())

    for classe in CLASSES:
        coluna = f"score_{classe}"

        assert coluna in resultado.columns


def test_scores_normalizados_somam_um(modelo_mock):
    resultado = prever(criar_entrada())

    colunas_scores = [
        f"score_{classe}"
        for classe in CLASSES
    ]

    soma = resultado[colunas_scores].sum(axis=1)

    assert np.allclose(soma, 1.0)


def test_prever_varios_registros(modelo_mock):
    entrada = pd.concat(
        [criar_entrada()] * 3,
        ignore_index=True,
    )

    resultado = prever(entrada)

    assert len(resultado) == 3


def test_prever_rejeita_dataframe_vazio(modelo_mock):
    entrada = pd.DataFrame(columns=FEATURES)

    with pytest.raises(ValueError, match="vazio"):
        prever(entrada)


def test_prever_rejeita_entrada_invalida(modelo_mock):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        prever(None)


def test_prever_identifica_features_ausentes(modelo_mock):
    entrada = criar_entrada().drop(
        columns=["latitude"]
    )

    with pytest.raises(ValueError, match="ausentes"):
        prever(entrada)
