"""Experimento: compara A*, custo uniforme (Dijkstra) e BFS em todas as origens/materiais.

Saídas: resultados/busca_comparacao.csv e resultados/graficos/busca_nos_expandidos.png

Uso:
    python -m src.decisao.experimento_busca
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import GRAFICOS, RESULTADOS
from src.decisao.grafo import MapaColeta

MATERIAIS = ["cardboard", "paper", "plastic", "metal", "glass"]


def rodar() -> pd.DataFrame:
    mapa = MapaColeta()
    linhas = []
    for origem in mapa.coord:
        for mat in MATERIAIS:
            for nome, fn in [("A*", mapa.a_estrela), ("Custo uniforme", mapa.custo_uniforme),
                             ("BFS", mapa.bfs)]:
                r = fn(origem, mat)
                linhas.append({"algoritmo": nome, "origem": origem, "material": mat,
                               "distancia_km": r.distancia_km, "nos_expandidos": r.nos_expandidos,
                               "bairros_no_caminho": len(r.caminho)})
    df = pd.DataFrame(linhas)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTADOS / "busca_comparacao.csv", index=False)

    resumo = df.groupby("algoritmo").agg(
        nos_expandidos_medio=("nos_expandidos", "mean"),
        distancia_media_km=("distancia_km", "mean"),
    )
    otimo = df[df.algoritmo == "Custo uniforme"].set_index(["origem", "material"])["distancia_km"]
    for alg in ["A*", "BFS"]:
        d = df[df.algoritmo == alg].set_index(["origem", "material"])["distancia_km"]
        resumo.loc[alg, "pct_caminho_otimo"] = ((d - otimo).abs() < 1e-6).mean() * 100
    resumo.loc["Custo uniforme", "pct_caminho_otimo"] = 100.0
    print(resumo.round(2).to_string())
    resumo.round(3).to_csv(RESULTADOS / "busca_resumo.csv")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ordem = ["BFS", "Custo uniforme", "A*"]
    ax.bar(ordem, [resumo.loc[a, "nos_expandidos_medio"] for a in ordem], color="#4C78A8")
    ax.set_ylabel("Nós expandidos (média)")
    ax.set_title("Esforço de busca por algoritmo")
    fig.tight_layout()
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAFICOS / "busca_nos_expandidos.png", dpi=200)
    return df


if __name__ == "__main__":
    rodar()
