from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

DATASETS_DIR = PROCESSED_DIR / "datasets"
MODELAGEM_DIR = PROCESSED_DIR / "modelagem"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

RESULTADOS_DIR = ROOT_DIR / "resultados"
REPORTS_DIR = ROOT_DIR / "reports"
