import json

import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from br_risk_ml.utils.paths import (
    ROOT_DIR,
    MODELAGEM_DIR,
    MODELS_DIR,
)

from br_risk_ml.models.pipeline import criar_preprocessador


CAMINHO_CONFIG = ROOT_DIR / "configs" / "model_config.json"


def carregar_configuracao():
    if not CAMINHO_CONFIG.exists():
        raise FileNotFoundError(
            f"Configuração não encontrada: {CAMINHO_CONFIG}"
        )

    with open(CAMINHO_CONFIG, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_dados(nome_arquivo, categoricas):
    caminho = MODELAGEM_DIR / nome_arquivo

    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho}"
        )

    df = pd.read_csv(
        caminho,
        sep=";",
        decimal=",",
        low_memory=False,
    )

    target = carregar_configuracao()["target"]

    df = df.dropna(subset=[target]).reset_index(drop=True)

    for coluna in categoricas:
        df[coluna] = (
            df[coluna]
            .fillna("Não informado")
            .astype(str)
        )

    return df


def treinar():
    print("=" * 55)
    print("BR-RISK-ML | TREINAMENTO")
    print("=" * 55)

    config = carregar_configuracao()
    categoricas = config["features"]["categorical"]
    numericas = config["features"]["numerical"]
    features = categoricas + numericas

    target = config["target"]
    classes_configuradas = config["classes"]

    params = config["xgboost"]
    expoente_pesos = config["training"]["sample_weight_exponent"]
    semente = config["training"]["random_state"]

    pesos_pos_treino = config["post_training_weights"]

    print("\nCarregando datasets...")

    treino = carregar_dados(
        "treino_expandido.csv",
        categoricas,
    )

    validacao = carregar_dados(
        "validacao_expandido.csv",
        categoricas,
    )

    dados = pd.concat(
        [treino, validacao],
        ignore_index=True,
    )

    print(f"Registros de treinamento: {len(dados):,}")

    X = dados[features]
    y_texto = dados[target]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_texto)

    classes = list(encoder.classes_)

    if classes != classes_configuradas:
        raise ValueError(
            "As classes encontradas nos dados são diferentes "
            "das classes definidas em model_config.json.\n"
            f"Configuração: {classes_configuradas}\n"
            f"Dados: {classes}"
        )

    print("Classes:", classes)

    pesos_amostras = compute_sample_weight(
        class_weight="balanced",
        y=y,
    ) ** expoente_pesos

    print("\nCriando pipeline...")

    preprocessador = criar_preprocessador(
        numericas=numericas,
        categoricas=categoricas,
    )

    modelo = XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=semente,
        n_jobs=-1,
        **params,
    )

    pipeline = Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", modelo),
    ])

    print("Treinando modelo...")

    pipeline.fit(
        X,
        y,
        modelo__sample_weight=pesos_amostras,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artefato = {
        "pipeline": pipeline,
        "classes": classes,
        "features": features,
        "pesos_pos_treino": pesos_pos_treino,
        "params_xgboost": params,
        "expoente_pesos_treino": expoente_pesos,
        "random_state": semente,
        "model_config": config,
        "features_categoricas": categoricas,
        "features_numericas": numericas,
    }

    caminho_modelo = MODELS_DIR / "br_risk_xgboost.joblib"

    joblib.dump(
        artefato,
        caminho_modelo,
    )

    print("\n" + "=" * 55)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 55)
    print(f"Registros utilizados: {len(dados):,}")
    print(f"Features: {len(features)}")
    print(f"Período: {config['training']['period']}")
    print(f"Modelo salvo em: {caminho_modelo}")


if __name__ == "__main__":
    treinar()
