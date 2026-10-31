import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from br_risk_ml.utils.paths import MODELAGEM_DIR, MODELS_DIR, REPORTS_DIR


CAMINHO_MODELO = MODELS_DIR / "br_risk_xgboost.joblib"

CAT = [
    "tipo_acidente",
    "causa_acidente",
    "fase_dia",
]

TARGET = "target"


def carregar_dados(nome_arquivo):
    df = pd.read_csv(
        MODELAGEM_DIR / nome_arquivo,
        sep=";",
        decimal=",",
        low_memory=False,
    )

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    for coluna in CAT:
        df[coluna] = (
            df[coluna]
            .fillna("Não informado")
            .astype(str)
        )

    return df


def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(CAMINHO_MODELO)


def avaliar(nome_arquivo):
    print(f"Carregando conjunto: {nome_arquivo}")

    dados = carregar_dados(nome_arquivo)
    artefato = carregar_modelo()

    pipeline = artefato["pipeline"]
    classes = artefato["classes"]
    features = artefato["features"]
    pesos = artefato["pesos_pos_treino"]

    X = dados[features]
    y_real = dados[TARGET]

    encoder_classes = {
        classe: indice
        for indice, classe in enumerate(classes)
    }

    y_real_encoded = y_real.map(encoder_classes)

    if y_real_encoded.isna().any():
        raise ValueError(
            "O conjunto possui classes que não existem no modelo."
        )

    y_real_encoded = y_real_encoded.astype(int).to_numpy()

    probabilidades = pipeline.predict_proba(X)

    multiplicadores = np.array([
        pesos.get(classe, pesos["demais"])
        for classe in classes
    ])

    probabilidades_ajustadas = probabilidades * multiplicadores
    y_pred = np.argmax(probabilidades_ajustadas, axis=1)

    accuracy = accuracy_score(y_real_encoded, y_pred)

    f1_macro = f1_score(
        y_real_encoded,
        y_pred,
        average="macro",
    )

    balanced_accuracy = balanced_accuracy_score(
        y_real_encoded,
        y_pred,
    )

    relatorio = classification_report(
        y_real_encoded,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    matriz = confusion_matrix(
        y_real_encoded,
        y_pred,
        labels=np.arange(len(classes)),
    )

    print("\n" + "=" * 55)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("=" * 55)

    print(f"Registros:          {len(dados):,}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"F1 Macro:           {f1_macro:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")

    print("\nMétricas por classe:")

    print(
        classification_report(
            y_real_encoded,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    print("Matriz de confusão:")
    print(
        pd.DataFrame(
            matriz,
            index=classes,
            columns=classes,
        )
    )

    return {
        "conjunto": nome_arquivo,
        "registros": len(dados),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "balanced_accuracy": float(balanced_accuracy),
        "por_classe": relatorio,
        "matriz_confusao": matriz.tolist(),
    }


if __name__ == "__main__":
    resultado = avaliar("validacao_expandido.csv")

    diretorio_saida = REPORTS_DIR / "avaliacao"
    diretorio_saida.mkdir(parents=True, exist_ok=True)

    caminho_saida = diretorio_saida / "metricas_validacao.json"

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(
            resultado,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nMétricas salvas em: {caminho_saida}")
