"""Configurações centrais do projeto (caminhos, classes e limiares)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
GRAFICOS = RESULTADOS / "graficos"
EXEMPLOS = RESULTADOS / "exemplos"

# Dataset de imagens (TrashNet)
DIR_IMAGENS = DADOS / "trashnet"          # dados/trashnet/<classe>/*.jpg
DIR_SPLITS = DADOS / "splits"             # CSVs de treino/validação/teste
SEMENTE = 42

# Classes do TrashNet (nomes das pastas) -> nome em português
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASSES_PT = {
    "cardboard": "papelão",
    "glass": "vidro",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "trash": "rejeito (não reciclável)",
}

# Modelos salvos
MODELO_CNN = MODELOS / "cnn_mobilenetv2.pt"
MODELO_BASELINE = MODELOS / "baseline_svm.joblib"
MODELO_INTENCOES = MODELOS / "intencoes.joblib"

# Arquivos de dados do PLN e da decisão
CSV_INTENCOES = DADOS / "intencoes.csv"
JSON_MAPA = DADOS / "mapa_maceio.json"

# Limiares usados pelo módulo de decisão
LIMIAR_CONFIANCA = 0.60      # abaixo disso o agente pede uma nova foto
LIMIAR_MARGEM = 0.15         # top1 - top2 menor que isso => pergunta ao usuário
LIMIAR_INTENCAO = 0.35       # abaixo disso a pergunta é tratada como fora de escopo
