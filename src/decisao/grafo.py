"""Busca A* no grafo de bairros de Maceió até o ponto de coleta mais próximo.

- Custo g(n): soma das distâncias das vias (haversine entre centros de bairro x fator viário).
- Heurística h(n): menor distância em linha reta (haversine) de n até QUALQUER bairro-objetivo
  que tenha ponto de coleta aceitando o material. Como o fator viário é >= 1, a linha reta
  nunca superestima o custo real => h é ADMISSÍVEL e o A* encontra o caminho ótimo.
- Também há BFS (menor número de bairros) e Dijkstra/custo uniforme (A* com h=0) para
  comparação no artigo (nós expandidos x otimalidade).
"""
from __future__ import annotations

import heapq
import json
import math
from collections import deque
from dataclasses import dataclass, field

from src.config import JSON_MAPA


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    d = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(d))


@dataclass
class ResultadoBusca:
    encontrado: bool
    caminho: list[str] = field(default_factory=list)
    distancia_km: float = 0.0
    nos_expandidos: int = 0
    ponto: dict | None = None


class MapaColeta:
    def __init__(self, caminho=JSON_MAPA):
        d = json.loads(caminho.read_text(encoding="utf-8"))
        self.coord = {b: tuple(c) for b, c in d["bairros"].items()}
        self.fator = d.get("fator_viario", 1.3)
        self.pontos = d["pontos_coleta"]
        self.vizinhos: dict[str, dict[str, float]] = {b: {} for b in self.coord}
        for a, b in d["vias"]:
            custo = haversine_km(self.coord[a], self.coord[b]) * self.fator
            self.vizinhos[a][b] = custo
            self.vizinhos[b][a] = custo

    def objetivos(self, material: str) -> dict[str, dict]:
        """Bairros que têm ponto de coleta aceitando o material -> ponto."""
        return {p["bairro"]: p for p in self.pontos if material in p["aceita"]}

    def _h(self, no: str, objetivos) -> float:
        return min(haversine_km(self.coord[no], self.coord[o]) for o in objetivos)

    def a_estrela(self, origem: str, material: str, usar_heuristica: bool = True) -> ResultadoBusca:
        objetivos = self.objetivos(material)
        if origem not in self.coord or not objetivos:
            return ResultadoBusca(False)
        h = (lambda n: self._h(n, objetivos)) if usar_heuristica else (lambda n: 0.0)
        fronteira = [(h(origem), 0.0, origem, [origem])]
        melhor_g = {origem: 0.0}
        expandidos = 0
        while fronteira:
            f, g, no, caminho = heapq.heappop(fronteira)
            if g > melhor_g.get(no, math.inf):
                continue  # entrada desatualizada
            expandidos += 1
            if no in objetivos:
                return ResultadoBusca(True, caminho, round(g, 2), expandidos, objetivos[no])
            for viz, custo in self.vizinhos[no].items():
                ng = g + custo
                if ng < melhor_g.get(viz, math.inf):
                    melhor_g[viz] = ng
                    heapq.heappush(fronteira, (ng + h(viz), ng, viz, caminho + [viz]))
        return ResultadoBusca(False, nos_expandidos=expandidos)

    def custo_uniforme(self, origem: str, material: str) -> ResultadoBusca:
        return self.a_estrela(origem, material, usar_heuristica=False)

    def bfs(self, origem: str, material: str) -> ResultadoBusca:
        """Menor número de bairros percorridos (ignora distância)."""
        objetivos = self.objetivos(material)
        if origem not in self.coord or not objetivos:
            return ResultadoBusca(False)
        fila, visitados, expandidos = deque([[origem]]), {origem}, 0
        while fila:
            caminho = fila.popleft()
            expandidos += 1
            if caminho[-1] in objetivos:
                dist = sum(self.vizinhos[a][b] for a, b in zip(caminho, caminho[1:]))
                return ResultadoBusca(True, caminho, round(dist, 2), expandidos, objetivos[caminho[-1]])
            for viz in self.vizinhos[caminho[-1]]:
                if viz not in visitados:
                    visitados.add(viz)
                    fila.append(caminho + [viz])
        return ResultadoBusca(False, nos_expandidos=expandidos)
