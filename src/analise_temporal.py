
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

ARQUIVO = (
    ROOT_DIR
    / "data"
    / "processed"
    / "acidentes_prf.csv"
)

df = pd.read_csv(
    ARQUIVO,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

# Extrai o ano da data do acidente
df["ano"] = pd.to_datetime(
    df["data_inversa"],
    errors="coerce"
).dt.year

# Remove registros sem classificação apenas desta análise
analise = df.dropna(
    subset=["classificacao_acidente"]
)

# Quantidade de acidentes por ano e classificação
tabela = pd.crosstab(
    analise["ano"],
    analise["classificacao_acidente"]
)

print("\nACIDENTES POR ANO E CLASSIFICAÇÃO")
print(tabela.to_string())

# Percentual de cada classificação dentro de cada ano
percentuais = pd.crosstab(
    analise["ano"],
    analise["classificacao_acidente"],
    normalize="index"
) * 100

print("\nPERCENTUAL POR ANO")
print(percentuais.round(2).to_string())

# Quantidade total por ano
print("\nTOTAL POR ANO")
print(analise.groupby("ano").size())
