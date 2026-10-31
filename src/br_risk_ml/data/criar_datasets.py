import pandas as pd

from br_risk_ml.utils.paths import INTERIM_DIR, DATASETS_DIR


ENTRADA = INTERIM_DIR / "acidentes_prf_preparado.csv"
SAIDA_DIR = DATASETS_DIR

FEATURES_BASICAS = [
    "uf",
    "br",
    "km",
    "municipio",
    "dia_semana",
    "hora",
    "mes",
    "condicao_metereologica",
    "tipo_pista",
    "uso_solo",
    "fase_dia",
    "sentido_via",
    "latitude",
    "longitude",
]

FEATURES_EXPANDIDAS = FEATURES_BASICAS + [
    "tipo_acidente",
    "causa_acidente",
    "pessoas",
    "veiculos",
]


def main() -> None:
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ENTRADA, sep=";", encoding="utf-8", low_memory=False)

    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")

    df["ano"] = df["data_inversa"].dt.year
    df["mes"] = df["data_inversa"].dt.month

    df["hora"] = pd.to_numeric(
        df["horario"].astype("string").str.extract(r"^(\d{1,2})")[0],
        errors="coerce",
    )

    for coluna in ["km", "latitude", "longitude"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df["br"] = df["br"].astype("string")

    quantidade_inicial = len(df)
    df = df.dropna(subset=["classificacao_acidente"]).copy()

    print("Registros iniciais:", quantidade_inicial)
    print("Registros sem target removidos:", quantidade_inicial - len(df))

    y = df["classificacao_acidente"]

    dataset_basico = df[FEATURES_BASICAS].copy()
    dataset_basico["target"] = y
    dataset_basico["ano"] = df["ano"]

    dataset_expandido = df[FEATURES_EXPANDIDAS].copy()
    dataset_expandido["target"] = y
    dataset_expandido["ano"] = df["ano"]

    dataset_basico.to_csv(
        SAIDA_DIR / "dataset_basico.csv",
        sep=";",
        index=False,
        encoding="utf-8",
    )

    dataset_expandido.to_csv(
        SAIDA_DIR / "dataset_expandido.csv",
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print("\nDISTRIBUIÇÃO DO TARGET:")
    print(y.value_counts())

    print("\nDATASET BÁSICO:")
    print("Registros:", dataset_basico.shape[0], "| Features:", len(FEATURES_BASICAS))

    print("\nDATASET EXPANDIDO:")
    print("Registros:", dataset_expandido.shape[0], "| Features:", len(FEATURES_EXPANDIDAS))

    print("\nDatasets salvos com sucesso!")


if __name__ == "__main__":
    main()
