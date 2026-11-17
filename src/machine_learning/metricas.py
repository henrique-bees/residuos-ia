"""Métricas e gráficos compartilhados (Visão, baseline e PLN)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import GRAFICOS, RESULTADOS


def relatorio(y_true, y_pred, nomes: list[str], prefixo: str) -> dict:
    """Calcula métricas, salva JSON e matriz de confusão em resultados/."""
    rotulos = list(range(len(nomes))) if isinstance(y_true[0], (int, np.integer)) else nomes
    rel = classification_report(
        y_true, y_pred, labels=rotulos, target_names=nomes, output_dict=True, zero_division=0
    )
    resumo = {
        "acuracia": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=rotulos, average="macro", zero_division=0),
        "por_classe": {n: rel[n] for n in nomes},
    }
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / f"metricas_{prefixo}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matriz_confusao(y_true, y_pred, rotulos, nomes, GRAFICOS / f"confusao_{prefixo}.png")
    print(f"[{prefixo}] acurácia={resumo['acuracia']:.3f}  F1-macro={resumo['f1_macro']:.3f}")
    return resumo


def matriz_confusao(y_true, y_pred, rotulos, nomes, destino: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=rotulos)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(nomes)), nomes, rotation=45, ha="right")
    ax.set_yticks(range(len(nomes)), nomes)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def curvas_treino(historico: list[dict], destino: Path) -> None:
    """Perda e acurácia por época (treino x validação)."""
    ep = [h["epoca"] for h in historico]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.plot(ep, [h["perda_treino"] for h in historico], label="treino")
    a1.plot(ep, [h["perda_val"] for h in historico], label="validação")
    a1.set_title("Perda"); a1.set_xlabel("época"); a1.legend()
    a2.plot(ep, [h["acc_treino"] for h in historico], label="treino")
    a2.plot(ep, [h["acc_val"] for h in historico], label="validação")
    a2.set_title("Acurácia"); a2.set_xlabel("época"); a2.legend()
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=200)
    plt.close(fig)
