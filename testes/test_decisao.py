import pytest

from src.decisao.agente import AgenteDescarte
from src.decisao.grafo import MapaColeta, haversine_km
from src.visao_computacional.classificador import ResultadoVisao

MAPA = MapaColeta()


def visao(classe, conf, segunda="paper"):
    resto = (1 - conf)
    probs = {classe: conf, segunda: resto * 0.9}
    probs.update({c: resto * 0.02 for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
                  if c not in probs})
    return ResultadoVisao(classe, conf, dict(sorted(probs.items(), key=lambda kv: -kv[1])))


# ---------- Busca ----------
@pytest.mark.parametrize("origem", list(MAPA.coord))
@pytest.mark.parametrize("material", ["glass", "metal", "paper"])
def test_a_estrela_otimo_igual_custo_uniforme(origem, material):
    a = MAPA.a_estrela(origem, material)
    u = MAPA.custo_uniforme(origem, material)
    assert a.encontrado and u.encontrado
    assert a.distancia_km == pytest.approx(u.distancia_km)
    assert a.nos_expandidos <= u.nos_expandidos


def test_heuristica_admissivel():
    """h(n) nunca supera o custo real até o objetivo mais próximo."""
    for material in ["glass", "metal", "plastic"]:
        objs = MAPA.objetivos(material)
        for no in MAPA.coord:
            real = MAPA.custo_uniforme(no, material).distancia_km
            assert MAPA._h(no, objs) <= real + 1e-9


def test_origem_ja_e_objetivo():
    r = MAPA.a_estrela("Ponta Verde", "glass")
    assert r.caminho == ["Ponta Verde"] and r.distancia_km == 0


def test_bfs_menos_bairros():
    b = MAPA.bfs("Trapiche da Barra", "glass")
    a = MAPA.a_estrela("Trapiche da Barra", "glass")
    assert len(b.caminho) <= len(a.caminho)
    assert b.distancia_km >= a.distancia_km - 1e-9


def test_haversine():
    assert haversine_km((0, 0), (0, 1)) == pytest.approx(111.19, rel=1e-3)


# ---------- Regras do agente ----------
AG = AgenteDescarte(mapa=MAPA)


def test_confianca_baixa_pede_nova_foto():
    r = AG.decidir(visao("glass", 0.40), "reciclavel", "isso recicla?")
    assert r.precisa_nova_foto and any(x.startswith("R6") for x in r.regras)


def test_texto_tem_prioridade_sobre_visao():
    r = AG.decidir(visao("glass", 0.95), "cor_lixeira", "essa lata vai em qual lixeira?")
    assert r.material == "metal" and r.fonte_material == "texto"
    assert "amarela" in r.texto and "R4" in " ".join(r.regras)


def test_rejeito_nao_reciclavel():
    r = AG.decidir(visao("trash", 0.9), "reciclavel", "pode reciclar?")
    assert r.texto.startswith("Não")


def test_onde_descartar_usa_a_estrela_com_bairro_do_texto():
    r = AG.decidir(visao("glass", 0.9), "onde_descartar", "estou no farol, onde levo?")
    assert r.rota[0] == "Farol" and "R15" in " ".join(r.regras)


def test_onde_descartar_sem_bairro_pergunta():
    r = AG.decidir(visao("metal", 0.9), "onde_descartar", "onde levo isso?")
    assert r.rota is None and "bairro" in r.texto


def test_sem_imagem_sem_material():
    r = AG.decidir(None, "reciclavel", "isso recicla?")
    assert r.precisa_nova_foto


def test_margem_pequena_avisa():
    probs = {"paper": 0.62, "cardboard": 0.52, "plastic": 0.0}  # margem 0.10 < 0.15
    r = AG.decidir(ResultadoVisao("paper", 0.62, probs), "identificar", "o que é?")
    assert "papelão" in r.texto and "R8" in " ".join(r.regras)


def test_material_lembrado_eh_usado_quando_pergunta_atual_nao_cita_material():
    """Reaproveita o material de uma mensagem anterior (ex.: 'isso é plástico' / depois
    'como reciclo isso?') sem precisar repeti-lo na mesma mensagem."""
    r = AG.decidir(None, "como_descartar", "como faço para reciclar isso?",
                   material_lembrado="plastic")
    assert r.material == "plastic" and r.fonte_material == "texto"
    assert any(x.startswith("R3b") for x in r.regras)


def test_material_da_pergunta_atual_tem_prioridade_sobre_lembrado():
    r = AG.decidir(None, "reciclavel", "essa lata é reciclável?", material_lembrado="glass")
    assert r.material == "metal"


def test_onde_descartar_sem_bairro_marca_aguardando_bairro():
    r = AG.decidir(visao("metal", 0.9), "onde_descartar", "onde levo isso?")
    assert r.aguardando_bairro is True


def test_identificar_nao_marca_aguardando_bairro():
    r = AG.decidir(visao("metal", 0.9), "identificar", "o que é isso?")
    assert r.aguardando_bairro is False
