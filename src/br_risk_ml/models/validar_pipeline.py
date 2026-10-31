import numpy as np
import pandas as pd

from br_risk_ml.models.pipeline import (
    criar_preprocessador,
    obter_variaveis,
)

from br_risk_ml.utils.paths import MODELAGEM_DIR


def testar_preprocessamento(nome):
    treino = pd.read_csv(
        MODELAGEM_DIR / f"treino_{nome}.csv",
        sep=";",
        encoding="utf-8",
    )

    validacao = pd.read_csv(
        MODELAGEM_DIR / f"validacao_{nome}.csv",
        sep=";",
        encoding="utf-8",
    )

    teste = pd.read_csv(
        MODELAGEM_DIR / f"teste_{nome}.csv",
        sep=";",
        encoding="utf-8",
    )

    numericas, categoricas = obter_variaveis(nome)

    X_treino = treino.drop(columns=["target"])
    y_treino = treino["target"]

    X_validacao = validacao.drop(columns=["target"])
    X_teste = teste.drop(columns=["target"])

    for coluna in categoricas:
        for conjunto in [X_treino, X_validacao, X_teste]:
            conjunto[coluna] = conjunto[coluna].map(
                lambda valor: str(valor)
                if pd.notna(valor)
                else np.nan
            )

    preprocessador = criar_preprocessador(
        numericas,
        categoricas,
    )

    X_treino_transformado = preprocessador.fit_transform(
        X_treino
    )

    X_validacao_transformado = preprocessador.transform(
        X_validacao
    )

    X_teste_transformado = preprocessador.transform(
        X_teste
    )

    print("\n" + "=" * 60)
    print(f"PREPROCESSAMENTO: {nome.upper()}")
    print("=" * 60)

    print("\nFormato original:")
    print(f"Treinamento: {X_treino.shape}")
    print(f"Validação: {X_validacao.shape}")
    print(f"Teste: {X_teste.shape}")

    print("\nFormato transformado:")
    print(f"Treinamento: {X_treino_transformado.shape}")
    print(f"Validação: {X_validacao_transformado.shape}")
    print(f"Teste: {X_teste_transformado.shape}")

    print("\nFormato esparso?")
    print(hasattr(X_treino_transformado, "toarray"))

    print("\nQuantidade de classes do target:")
    print(y_treino.nunique())

    print("\nPré-processamento concluído!")


if __name__ == "__main__":
    testar_preprocessamento("basico")
    testar_preprocessamento("expandido")
