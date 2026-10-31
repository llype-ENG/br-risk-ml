import pandas as pd

from br_risk_ml.utils.paths import DATASETS_DIR, MODELAGEM_DIR


ANOS_TREINO = list(range(2017, 2023))
ANOS_VALIDACAO = [2023]
ANOS_TESTE = [2024, 2025]


def separar_dataset(nome: str) -> None:
    df = pd.read_csv(
        DATASETS_DIR / f"dataset_{nome}.csv",
        sep=";",
        encoding="utf-8",
    )

    treino = df[df["ano"].isin(ANOS_TREINO)].drop(columns=["ano"])
    validacao = df[df["ano"].isin(ANOS_VALIDACAO)].drop(columns=["ano"])
    teste = df[df["ano"].isin(ANOS_TESTE)].drop(columns=["ano"])

    print("\n" + "=" * 60)
    print(f"DATASET {nome.upper()}")
    print("=" * 60)

    for nome_conjunto, dados in {
        "TREINAMENTO": treino,
        "VALIDAÇÃO": validacao,
        "TESTE": teste,
    }.items():
        print(f"\n{nome_conjunto}")
        print(f"Registros: {len(dados)}")
        print("Distribuição do target (%):")
        print((dados["target"].value_counts(normalize=True) * 100).round(2))

    treino.to_csv(
        MODELAGEM_DIR / f"treino_{nome}.csv",
        sep=";",
        index=False,
        encoding="utf-8",
    )

    validacao.to_csv(
        MODELAGEM_DIR / f"validacao_{nome}.csv",
        sep=";",
        index=False,
        encoding="utf-8",
    )

    teste.to_csv(
        MODELAGEM_DIR / f"teste_{nome}.csv",
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print("\nArquivos salvos com sucesso!")


def main() -> None:
    MODELAGEM_DIR.mkdir(parents=True, exist_ok=True)

    separar_dataset("basico")
    separar_dataset("expandido")

    print("\nSeparação temporal concluída!")


if __name__ == "__main__":
    main()
