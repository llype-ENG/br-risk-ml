from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARQUIVO_SAIDA = PROCESSED_DIR / "acidentes_prf.csv"

PADRAO_ARQUIVOS = "*.csv"
ARQUIVOS_ESPERADOS = 9  # 2017 a 2025


def carregar_csv(arquivo: Path) -> pd.DataFrame:
    """Tenta carregar um CSV usando diferentes encodings."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(
                arquivo,
                sep=";",
                encoding=encoding,
                low_memory=False,
            )
            print(f"  Encoding utilizado: {encoding}")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Não foi possível identificar o encoding de {arquivo.name}")


def localizar_arquivos() -> list[Path]:
    arquivos = sorted(RAW_DIR.glob(PADRAO_ARQUIVOS))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em: {RAW_DIR}")

    if len(arquivos) != ARQUIVOS_ESPERADOS:
        print(
            f"ATENÇÃO: encontrei {len(arquivos)} arquivos, mas o esperado era "
            f"{ARQUIVOS_ESPERADOS}. Confira se há arquivos que não deveriam estar em data/raw."
        )

    return arquivos


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    arquivos = localizar_arquivos()

    print("=" * 55)
    print("CONSOLIDAÇÃO DOS DADOS DA PRF")
    print("=" * 55)

    dados = []
    for arquivo in arquivos:
        print(f"\nCarregando: {arquivo.name}")
        df = carregar_csv(arquivo)
        df["arquivo_origem"] = arquivo.stem
        dados.append(df)
        print(f"  Registros: {len(df):,}")
        print(f"  Colunas: {len(df.columns):,}")

    print("\nConcatenando os arquivos...")
    dataset = pd.concat(dados, ignore_index=True, sort=False)

    dataset.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8", sep=";")

    print("\n" + "=" * 55)
    print("CONSOLIDAÇÃO CONCLUÍDA")
    print("=" * 55)
    print(f"Arquivos processados: {len(arquivos)}")
    print(f"Total de registros: {len(dataset):,}")
    print(f"Total de colunas: {len(dataset.columns):,}")
    print(f"\nArquivo salvo em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()
