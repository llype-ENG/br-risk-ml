import joblib
import pandas as pd

from br_risk_ml.utils.paths import MODELS_DIR


CAMINHO_MODELO = MODELS_DIR / "br_risk_xgboost.joblib"


def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(CAMINHO_MODELO)


def prever(dados: pd.DataFrame) -> pd.DataFrame:
    artefato = carregar_modelo()

    pipeline = artefato["pipeline"]
    classes = artefato["classes"]
    features = artefato["features"]
    categoricas = artefato["features_categoricas"]
    pesos = artefato["pesos_pos_treino"]

    if not isinstance(dados, pd.DataFrame):
        raise TypeError("A entrada deve ser um pandas DataFrame.")

    if dados.empty:
        raise ValueError("O DataFrame de entrada está vazio.")

    faltantes = [
        coluna for coluna in features
        if coluna not in dados.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {faltantes}"
        )

    X = dados[features].copy()

    for coluna in categoricas:
        X[coluna] = (
            X[coluna]
            .fillna("Não informado")
            .astype(str)
        )

    probabilidades = pipeline.predict_proba(X)

    multiplicadores = [
        pesos.get(classe, pesos["demais"])
        for classe in classes
    ]

    scores = probabilidades * multiplicadores
    scores_normalizados = scores / scores.sum(axis=1, keepdims=True)

    indices = scores_normalizados.argmax(axis=1)

    resultado = pd.DataFrame({
        "classe_prevista": [
            classes[indice] for indice in indices
        ]
    })

    for indice, classe in enumerate(classes):
        resultado[f"score_{classe}"] = (
            scores_normalizados[:, indice]
        )

    return resultado


if __name__ == "__main__":
    exemplo = pd.DataFrame([
        {
            "tipo_acidente": "Colisão traseira",
            "causa_acidente": "Falta de atenção à condução",
            "fase_dia": "Pleno dia",
            "pessoas": 2,
            "veiculos": 2,
            "latitude": -15.7801,
            "longitude": -47.9292,
        }
    ])

    resultado = prever(exemplo)

    print("\nResultado da previsão:")
    print(resultado.to_string(index=False))
