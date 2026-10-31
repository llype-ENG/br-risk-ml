
import pandas as pd
import pytest

from br_risk_ml.models.predict import (
    carregar_modelo,
    prever,
)


@pytest.fixture(scope="module")
def modelo():
    """Carrega o artefato real uma única vez."""
    return carregar_modelo()


@pytest.fixture
def entrada():
    """Registro fictício para inferência real."""

    return pd.DataFrame([{
        "tipo_acidente": "Colisão traseira",
        "causa_acidente": "Falta de atenção à condução",
        "fase_dia": "Pleno dia",
        "pessoas": 2,
        "veiculos": 2,
        "latitude": -15.7801,
        "longitude": -47.9292,
    }])


def test_artefato_possui_estrutura_esperada(modelo):
    chaves_obrigatorias = [
        "pipeline",
        "classes",
        "features",
        "features_categoricas",
        "features_numericas",
        "pesos_pos_treino",
    ]

    for chave in chaves_obrigatorias:
        assert chave in modelo


def test_pipeline_real_consegue_prever(modelo, entrada):
    probabilidades = modelo["pipeline"].predict_proba(
        entrada[modelo["features"]]
    )

    assert probabilidades.shape == (1, len(modelo["classes"]))


def test_modelo_real_retorna_previsao(entrada):
    resultado = prever(entrada)

    assert len(resultado) == 1
    assert "classe_prevista" in resultado.columns


def test_modelo_real_retorna_scores_validos(entrada):
    resultado = prever(entrada)

    colunas_scores = [
        coluna for coluna in resultado.columns
        if coluna.startswith("score_")
    ]

    assert len(colunas_scores) == 3
    assert resultado[colunas_scores].notna().all().all()
