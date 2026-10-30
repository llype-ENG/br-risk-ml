
import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ARQUIVO_SAIDA = PROCESSED_DIR / "acidentes_prf.csv"


def carregar_csv(arquivo):
    """
    Tenta carregar um CSV utilizando diferentes encodings.
    """

    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                arquivo,
                sep=";",
                encoding=encoding,
                low_memory=False
            )

            print(f"  Encoding utilizado: {encoding}")

            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Não foi possível identificar o encoding de {arquivo.name}"
    )

def localizar_arquivos():

    arquivos = sorted(RAW_DIR.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado em: {RAW_DIR}"
        )

    return arquivos


def processar_dados():

    arquivos = localizar_arquivos()

    dados = []

    print("=" * 55)
    print("INICIANDO PROCESSAMENTO DOS DADOS DA PRF")
    print("=" * 55)

    for arquivo in arquivos:

        print(f"\nCarregando: {arquivo.name}")

        df = carregar_csv(arquivo)

        # Registra a origem de cada registro
        df["arquivo_origem"] = arquivo.stem

        dados.append(df)

        print(f"  Registros: {len(df):,}")
        print(f"  Colunas: {len(df.columns):,}")

    print("\nConcatenando os arquivos...")

    dataset = pd.concat(
        dados,
        ignore_index=True,
        sort=False
    )


    dataset.to_csv(
        ARQUIVO_SAIDA,
        index=False,
        encoding="utf-8",
        sep=";"
    )


    print("\n" + "=" * 55)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("=" * 55)

    print(f"Arquivos processados: {len(arquivos)}")
    print(f"Total de registros: {len(dataset):,}")
    print(f"Total de colunas: {len(dataset.columns):,}")

    print(f"\nArquivo salvo em:")
    print(ARQUIVO_SAIDA)

    print("\nColunas identificadas:")
    print(dataset.columns.tolist())

    print("\nPrimeiros registros:")
    print(dataset.head())


if __name__ == "__main__":
    processar_dados()
