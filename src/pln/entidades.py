"""Extração de entidades da pergunta: material citado e bairro do usuário.

Abordagem baseada em léxico (dicionário de termos) sobre o texto normalizado.
Permite que o texto complemente ou corrija a visão: "isso é uma lata?" -> metal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.config import JSON_MAPA
from src.pln.preprocessamento import normalizar

# termo (normalizado) -> classe do TrashNet. Termos com mais palavras têm prioridade.
LEXICO_MATERIAIS = {
    "cardboard": ["papelao", "caixa de papelao", "caixa de sapato", "caixa de leite",
                  "embalagem de papelao", "caixa"],
    "glass": ["vidro", "garrafa de vidro", "pote de vidro", "copo de vidro", "vidro quebrado",
              "long neck", "frasco de vidro", "espelho"],
    "metal": ["metal", "lata", "latinha", "aluminio", "lata de refrigerante", "lata de cerveja",
              "tampinha", "ferro", "aco", "panela"],
    "paper": ["papel", "jornal", "revista", "folha", "caderno", "papel sulfite", "envelope",
              "panfleto", "livro"],
    "plastic": ["plastico", "pet", "garrafa pet", "sacola", "sacolinha", "copo descartavel",
                "embalagem plastica", "isopor", "pote de plastico", "canudo", "garrafa plastica"],
    "trash": ["guardanapo", "papel higienico", "fralda", "bituca", "esponja", "lixo comum",
              "rejeito", "papel engordurado", "embalagem metalizada", "salgadinho"],
}


@dataclass
class Entidades:
    material: str | None = None          # classe do TrashNet citada no texto
    termo_material: str | None = None    # termo encontrado (ex.: "garrafa pet")
    bairro: str | None = None
    termos: list[str] = field(default_factory=list)


def _contem(texto: str, termo: str) -> bool:
    return re.search(rf"\b{re.escape(termo)}\b", texto) is not None


def _carregar_bairros() -> dict[str, str]:
    dados = json.loads(JSON_MAPA.read_text(encoding="utf-8"))
    return {normalizar(b): b for b in dados["bairros"]}


_BAIRROS = _carregar_bairros()
# lista (termo, classe) ordenada do termo mais longo para o mais curto
_TERMOS = sorted(((t, c) for c, ts in LEXICO_MATERIAIS.items() for t in ts),
                 key=lambda tc: -len(tc[0]))


def extrair(texto: str) -> Entidades:
    t = normalizar(texto)
    ent = Entidades()
    for termo, classe in _TERMOS:
        if _contem(t, termo):
            ent.termos.append(termo)
            if ent.material is None:
                ent.material, ent.termo_material = classe, termo
    for b_norm, b in sorted(_BAIRROS.items(), key=lambda kv: -len(kv[0])):
        if _contem(t, b_norm):
            ent.bairro = b
            break
    return ent


def com_memoria_de_material(pergunta: str, termo_lembrado: str | None) -> str:
    """Reinsere, como contexto, o último material citado pelo usuário em mensagens anteriores
    da mesma conversa, quando a pergunta atual não cita nenhum material. Sem isso, uma conversa
    em turnos ("isso é plástico" / depois "como reciclo isso?") perde o material a cada mensagem,
    já que cada pergunta é processada isoladamente."""
    if termo_lembrado and extrair(pergunta).material is None:
        return f"(material mencionado antes: {termo_lembrado}) {pergunta}"
    return pergunta
