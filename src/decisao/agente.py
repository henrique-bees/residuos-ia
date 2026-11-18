"""Agente de descarte: integra Visão Computacional, PLN e decisão (regras + A*).

Ciclo do agente (perceber -> decidir -> agir):
  1. Percepção: resultado da CNN sobre a imagem + intenção e entidades da pergunta.
  2. Decisão: sistema de regras com prioridade resolve o MATERIAL (texto x visão x
     confiança) e depois a RESPOSTA conforme a intenção; para "onde descartar" aciona A*.
  3. Ação: resposta em linguagem natural + rastro das regras disparadas (explicabilidade).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import CLASSES_PT, LIMIAR_CONFIANCA, LIMIAR_MARGEM
from src.decisao.grafo import MapaColeta
from src.decisao.regras import CONHECIMENTO, EMOJI_COR
from src.pln import entidades as ent_mod


@dataclass
class Resposta:
    texto: str
    material: str | None = None
    fonte_material: str | None = None      # "texto", "visao" ou None
    intencao: str | None = None
    regras: list[str] = field(default_factory=list)
    rota: list[str] | None = None
    precisa_nova_foto: bool = False
    aguardando_bairro: bool = False


class AgenteDescarte:
    def __init__(self, classificador_visao=None, classificador_intencoes=None, mapa=None):
        self.visao = classificador_visao
        self.pln = classificador_intencoes
        self.mapa = mapa or MapaColeta()

    # ---------- interface usada pelo app ----------
    def responder(self, pergunta: str, imagem=None, bairro: str | None = None,
                  material_lembrado: str | None = None,
                  intencao_forcada: str | None = None) -> Resposta:
        r_visao = self.visao.classificar(imagem) if (imagem is not None and self.visao) else None
        intencao = intencao_forcada or self.pln.prever(pergunta).intencao
        return self.decidir(r_visao, intencao, pergunta, bairro, material_lembrado)

    # ---------- núcleo de decisão (testável sem modelos) ----------
    def decidir(self, r_visao, intencao: str, pergunta: str, bairro: str | None = None,
                material_lembrado: str | None = None) -> Resposta:
        regras: list[str] = []
        ent = ent_mod.extrair(pergunta)
        bairro = ent.bairro or bairro
        if ent.bairro:
            regras.append(f"R0: bairro extraído do texto -> {ent.bairro}")

        # Intenções que não dependem do material
        if intencao == "saudacao":
            regras.append("R1: saudação")
            return Resposta("Olá! Envie a foto de um resíduo e pergunte se é reciclável, "
                            "como descartar, a cor da lixeira ou onde levar.", intencao=intencao,
                            regras=regras)
        if intencao == "fora_escopo":
            regras.append("R2: fora de escopo")
            return Resposta("Só consigo ajudar com identificação e descarte de resíduos. "
                            "Tente algo como “isso é reciclável?” ou “onde levo isso?”.",
                            intencao=intencao, regras=regras)

        # ---- Resolução do material ----
        material, fonte, aviso = None, None, ""
        if ent.material:
            material, fonte = ent.material, "texto"
            regras.append(f"R3: material citado no texto (“{ent.termo_material}”) tem prioridade")
            if r_visao and r_visao.confianca >= LIMIAR_CONFIANCA and r_visao.classe != material:
                aviso = (f" (Obs.: pela imagem eu diria {CLASSES_PT[r_visao.classe]}, "
                         f"mas segui o que você escreveu.)")
                regras.append("R4: divergência texto x visão registrada")
        elif material_lembrado:
            material, fonte = material_lembrado, "texto"
            regras.append(f"R3b: material lembrado de mensagem anterior (“{material_lembrado}”) "
                          "tem prioridade")
            if r_visao and r_visao.confianca >= LIMIAR_CONFIANCA and r_visao.classe != material:
                aviso = (f" (Obs.: pela imagem eu diria {CLASSES_PT[r_visao.classe]}, "
                         f"mas segui o material que você disse antes.)")
                regras.append("R4: divergência texto x visão registrada")
        elif r_visao is None:
            regras.append("R5: sem imagem e sem material no texto")
            return Resposta("Não sei de qual resíduo você está falando. Envie uma foto ou "
                            "diga o material (ex.: “garrafa PET”, “lata”).", intencao=intencao,
                            regras=regras, precisa_nova_foto=True)
        elif r_visao.confianca < LIMIAR_CONFIANCA:
            regras.append(f"R6: confiança baixa ({r_visao.confianca:.0%} < {LIMIAR_CONFIANCA:.0%})")
            top2 = list(r_visao.probabilidades)[:2]
            return Resposta(
                f"Não tenho certeza do material (talvez {CLASSES_PT[top2[0]]} ou "
                f"{CLASSES_PT[top2[1]]}). Tire outra foto com o objeto centralizado, fundo "
                f"liso e boa iluminação, ou me diga o material.",
                intencao=intencao, regras=regras, precisa_nova_foto=True)
        else:
            material, fonte = r_visao.classe, "visao"
            regras.append(f"R7: material pela visão ({r_visao.confianca:.0%})")
            if r_visao.margem < LIMIAR_MARGEM:
                segunda = list(r_visao.probabilidades)[1]
                aviso = (f" (Atenção: também pode ser {CLASSES_PT[segunda]}; se for, me avise.)")
                regras.append(f"R8: margem pequena entre 1ª e 2ª classe ({r_visao.margem:.0%})")

        info = CONHECIMENTO[material]
        base = dict(material=material, fonte_material=fonte, intencao=intencao, regras=regras)
        conf = f" ({r_visao.confianca:.0%} de confiança)" if fonte == "visao" else ""

        # ---- Resposta conforme a intenção ----
        if intencao == "identificar":
            regras.append("R9: responder identificação")
            return Resposta(f"Parece ser **{info['nome']}**{conf}.{aviso}", **base)

        if intencao == "reciclavel":
            regras.append("R10: consultar base sobre reciclabilidade")
            if info["reciclavel"]:
                txt = f"Sim, **{info['nome']}** é reciclável — vai na coleta seletiva."
            else:
                txt = "Não. Isso é **rejeito** e deve ir no lixo comum (coleta domiciliar)."
            return Resposta(txt + aviso, **base)

        if intencao == "cor_lixeira":
            regras.append("R11: cor da lixeira (CONAMA 275/2001)")
            cor = info["lixeira"]
            return Resposta(f"{EMOJI_COR[cor]} Lixeira **{cor}** ({info['nome']}).{aviso}", **base)

        if intencao == "como_descartar":
            regras.append("R12: orientações de preparo")
            return Resposta(f"**{info['nome'].capitalize()}**: {info['preparo']}{aviso}", **base)

        if intencao == "onde_descartar":
            if not info["reciclavel"]:
                regras.append("R13: rejeito -> coleta domiciliar, sem busca")
                return Resposta("Rejeito não vai para ponto de coleta: descarte em saco fechado "
                                "na coleta domiciliar comum." + aviso, **base)
            if not bairro:
                regras.append("R14: falta a localização do usuário")
                return Resposta(f"Para achar o ponto mais próximo que recebe {info['nome']}, "
                                "me diga seu bairro.", aguardando_bairro=True, **base)
            busca = self.mapa.a_estrela(bairro, material)
            regras.append(f"R15: A* a partir de {bairro} ({busca.nos_expandidos} nós expandidos)")
            if not busca.encontrado:
                return Resposta(f"Não encontrei ponto que receba {info['nome']} a partir de "
                                f"{bairro}.", **base)
            rota = " → ".join(busca.caminho)
            return Resposta(
                f"O ponto mais próximo que recebe **{info['nome']}** é **{busca.ponto['nome']}**, "
                f"a cerca de {busca.distancia_km:.1f} km. Rota pelos bairros: {rota}.{aviso}",
                rota=busca.caminho, **base)

        regras.append("R16: intenção não tratada")
        return Resposta(f"Identifiquei {info['nome']}. Pergunte se é reciclável, como "
                        "descartar, a cor da lixeira ou onde levar.", **base)
