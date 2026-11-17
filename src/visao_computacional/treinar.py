"""Treino da CNN (MobileNetV2) em duas fases: extração de características e fine-tuning.

Uso:
    python -m src.visao_computacional.treinar --epocas 8 --epocas-ft 6
    python -m src.visao_computacional.treinar --amostra 60 --epocas 1 --epocas-ft 0   # teste rápido
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CLASSES, CLASSES_PT, MODELO_CNN, RESULTADOS, GRAFICOS, SEMENTE
from src.machine_learning.metricas import curvas_treino, relatorio
from src.visao_computacional.dataset import carregar_split
from src.visao_computacional.modelo import (
    TRANSF_AVAL, TRANSF_TREINO, DatasetResiduos, criar_modelo, liberar_ultimas_camadas,
)


def _amostrar(df, n):
    if not n:
        return df
    return df.groupby("classe", group_keys=False).apply(
        lambda g: g.sample(min(n, len(g)), random_state=SEMENTE)
    )


def _rodar_epoca(modelo, loader, criterio, otimizador, disp):
    treinando = otimizador is not None
    modelo.train(treinando)
    perda_total, acertos, n = 0.0, 0, 0
    with torch.set_grad_enabled(treinando):
        for x, y in loader:
            x, y = x.to(disp), y.to(disp)
            saida = modelo(x)
            perda = criterio(saida, y)
            if treinando:
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
            perda_total += perda.item() * len(y)
            acertos += (saida.argmax(1) == y).sum().item()
            n += len(y)
    return perda_total / n, acertos / n


@torch.no_grad()
def prever(modelo, loader, disp):
    modelo.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += modelo(x.to(disp)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return y_true, y_pred


def treinar(epocas=8, epocas_ft=6, lote=32, lr=1e-3, lr_ft=1e-4, amostra=0, pre_treinado=True):
    torch.manual_seed(SEMENTE)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {disp}")

    dfs = {n: _amostrar(carregar_split(n), amostra) for n in ("treino", "validacao", "teste")}
    lt = DataLoader(DatasetResiduos(dfs["treino"], TRANSF_TREINO), lote, shuffle=True, num_workers=2)
    lv = DataLoader(DatasetResiduos(dfs["validacao"], TRANSF_AVAL), lote, num_workers=2)
    lte = DataLoader(DatasetResiduos(dfs["teste"], TRANSF_AVAL), lote, num_workers=2)

    modelo = criar_modelo(pre_treinado=pre_treinado).to(disp)
    criterio = nn.CrossEntropyLoss()
    historico, melhor_acc = [], -1.0

    fases = [("extração", epocas, lr), ("fine-tuning", epocas_ft, lr_ft)]
    ep_global = 0
    for nome_fase, n_ep, taxa in fases:
        if n_ep == 0:
            continue
        if nome_fase == "fine-tuning":
            liberar_ultimas_camadas(modelo)
        otim = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad], lr=taxa)
        for _ in range(n_ep):
            ep_global += 1
            t0 = time.time()
            pt, at = _rodar_epoca(modelo, lt, criterio, otim, disp)
            pv, av = _rodar_epoca(modelo, lv, criterio, None, disp)
            historico.append({"epoca": ep_global, "fase": nome_fase, "perda_treino": pt,
                              "acc_treino": at, "perda_val": pv, "acc_val": av})
            print(f"[{nome_fase}] ép {ep_global}: perda {pt:.3f}/{pv:.3f}  "
                  f"acc {at:.3f}/{av:.3f}  ({time.time() - t0:.0f}s)")
            if av > melhor_acc:
                melhor_acc = av
                MODELO_CNN.parent.mkdir(parents=True, exist_ok=True)
                torch.save(modelo.state_dict(), MODELO_CNN)

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    (RESULTADOS / "historico_cnn.json").write_text(json.dumps(historico, indent=2), encoding="utf-8")
    curvas_treino(historico, GRAFICOS / "curvas_cnn.png")

    # Avaliação final no conjunto de TESTE com o melhor modelo (pela validação)
    modelo.load_state_dict(torch.load(MODELO_CNN, map_location=disp))
    y_true, y_pred = prever(modelo, lte, disp)
    return relatorio(y_true, y_pred, [CLASSES_PT[c] for c in CLASSES], "cnn")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--epocas-ft", type=int, default=6)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--amostra", type=int, default=0, help="imagens por classe (0 = todas)")
    ap.add_argument("--sem-pretreino", action="store_true")
    a = ap.parse_args()
    treinar(a.epocas, a.epocas_ft, a.lote, amostra=a.amostra, pre_treinado=not a.sem_pretreino)
