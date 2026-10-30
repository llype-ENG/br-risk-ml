
import pandas as pd
from pathlib import Path


# ==========================================
# 1. CONFIGURAÇÃO
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

ARQUIVO = ROOT_DIR / "data" / "processed" / "acidentes_prf.csv"


# ==========================================
# 2. CARREGAMENTO
# ==========================================

print("=" * 60)
print("ANÁLISE EXPLORATÓRIA - BR-RISK ML")
print("=" * 60)

df = pd.read_csv(
    ARQUIVO,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

print(f"\nTotal de registros: {len(df):,}")
print(f"Total de colunas: {len(df.columns)}")


# ==========================================
# 3. TIPOS DE DADOS
# ==========================================

print("\n" + "=" * 60)
print("TIPOS DE DADOS")
print("=" * 60)

print(df.dtypes)


# ==========================================
# 4. DISTRIBUIÇÃO POR ANO
# ==========================================

print("\n" + "=" * 60)
print("ACIDENTES POR ANO")
print("=" * 60)

print(
    df["arquivo_origem"]
    .value_counts()
    .sort_index()
)


# ==========================================
# 5. CLASSIFICAÇÃO DOS ACIDENTES
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICAÇÃO DOS ACIDENTES")
print("=" * 60)

print(
    df["classificacao_acidente"]
    .value_counts(dropna=False)
)


# ==========================================
# 6. VALORES AUSENTES
# ==========================================

print("\n" + "=" * 60)
print("VALORES AUSENTES")
print("=" * 60)

ausentes = pd.DataFrame({
    "quantidade": df.isnull().sum(),
    "percentual": df.isnull().mean() * 100
})

ausentes = ausentes.sort_values(
    by="percentual",
    ascending=False
)

print(ausentes.to_string())


# ==========================================
# 7. REGISTROS DUPLICADOS
# ==========================================

print("\n" + "=" * 60)
print("REGISTROS DUPLICADOS")
print("=" * 60)

print(f"Linhas totalmente duplicadas: {df.duplicated().sum():,}")

print(
    "IDs repetidos dentro da mesma origem:",
    df.duplicated(
        subset=["id", "arquivo_origem"]
    ).sum()
)


# ==========================================
# 8. AMOSTRA DOS DADOS
# ==========================================

print("\n" + "=" * 60)
print("AMOSTRA DOS DADOS")
print("=" * 60)

print(df.head(10).to_string())


# ==========================================
# 9. RESUMO ESTATÍSTICO
# ==========================================

print("\n" + "=" * 60)
print("RESUMO ESTATÍSTICO")
print("=" * 60)

print(df.describe(include="all").transpose())


print("\nANÁLISE FINALIZADA!")
