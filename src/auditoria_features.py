
from pathlib import Path
import pandas as pd

# Caminhos do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "processed" / "acidentes_prf.csv"

# Carregando os dados
df = pd.read_csv(DATA_FILE, sep=";", encoding="utf-8")

print("=" * 60)
print("AUDITORIA DE FEATURES - BR-RISK ML")
print("=" * 60)

# 1. Verificar distribuição das classes
print("\n1. DISTRIBUIÇÃO DO TARGET")

print(
    df["classificacao_acidente"]
    .value_counts(dropna=False)
)

# 2. Verificar quantidade de valores diferentes
print("\n2. QUANTIDADE DE VALORES ÚNICOS")

for coluna in [
    "tipo_acidente",
    "causa_acidente",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
    "fase_dia",
    "dia_semana",
]:
    print(f"\n{coluna}:")
    print("Valores únicos:", df[coluna].nunique())

    print(
        df[coluna]
        .value_counts()
        .head(15)
    )

# 3. Investigar valores numéricos e quantidade de envolvidos
print("\n3. DISTRIBUIÇÃO DE PESSOAS E VEÍCULOS")

print(
    df[["pessoas", "veiculos"]]
    .describe()
)

# 4. Investigar dados ausentes nas variáveis candidatas
print("\n4. VALORES AUSENTES")

print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

# 5. Investigar formatos de data e horário
print("\n5. EXEMPLOS DE DATA E HORÁRIO")

print("Datas:")
print(df["data_inversa"].head(10).to_list())

print("\nHorários:")
print(df["horario"].head(10).to_list())

# 6. Investigar coordenadas e quilometragem
print("\n6. EXEMPLOS DE LOCALIZAÇÃO")

for coluna in ["km", "latitude", "longitude"]:
    print(f"\n{coluna}:")
    print(df[coluna].head(10).to_list())

print("\nAuditoria concluída!")
