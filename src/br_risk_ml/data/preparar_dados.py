import pandas as pd

from br_risk_ml.utils.paths import INTERIM_DIR


ENTRADA = INTERIM_DIR / "acidentes_prf.csv"
SAIDA = INTERIM_DIR / "acidentes_prf_preparado.csv"

LATITUDE_LIMITES = (-36, 7)
LONGITUDE_LIMITES = (-76, -25)


def main() -> None:
    df = pd.read_csv(ENTRADA, sep=";", encoding="utf-8", low_memory=False)
    df_preparado = df.copy()

    for coluna in ["km", "latitude", "longitude"]:
        df_preparado[coluna] = pd.to_numeric(
            df_preparado[coluna].astype("string").str.replace(",", ".", regex=False),
            errors="coerce",
        )

    latitude_valida = df_preparado["latitude"].between(*LATITUDE_LIMITES)
    longitude_valida = df_preparado["longitude"].between(*LONGITUDE_LIMITES)

    coordenada_zero = (
        (df_preparado["latitude"] == 0)
        & (df_preparado["longitude"] == 0)
    )

    coordenada_invalida = ~latitude_valida | ~longitude_valida | coordenada_zero

    print("REGISTROS COM COORDENADAS INVÁLIDAS:")
    print(coordenada_invalida.sum())

    df_preparado.loc[
        coordenada_invalida,
        ["latitude", "longitude"]
    ] = float("nan")

    print("\nVALORES AUSENTES APÓS A LIMPEZA:")
    print(df_preparado[["km", "latitude", "longitude"]].isnull().sum())

    print("\nESTATÍSTICAS DAS COORDENADAS:")
    print(df_preparado[["latitude", "longitude"]].describe())

    df_preparado.to_csv(
        SAIDA,
        sep=";",
        index=False,
        encoding="utf-8",
    )

    print(f"\nArquivo preparado salvo em: {SAIDA}")


if __name__ == "__main__":
    main()
