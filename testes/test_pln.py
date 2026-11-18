import pytest

from src.config import MODELO_INTENCOES
from src.pln.entidades import extrair
from src.pln.preprocessamento import normalizar


def test_normalizar():
    assert normalizar("Dá pra RECICLAR, vc acha?") == "da pra reciclar voce acha"


@pytest.mark.parametrize("texto,material", [
    ("essa garrafa PET é reciclável?", "plastic"),
    ("onde jogo latinha", "metal"),
    ("papel engordurado recicla?", "trash"),      # termo mais longo vence "papel"
    ("caixa de papelão vai onde", "cardboard"),
    ("pote de vidro quebrou", "glass"),
    ("o que é isso?", None),
])
def test_extrair_material(texto, material):
    assert extrair(texto).material == material


def test_extrair_bairro():
    assert extrair("moro na ponta verde").bairro == "Ponta Verde"
    assert extrair("to no jatiuca agora").bairro == "Jatiúca"
    assert extrair("onde levo isso").bairro is None




@pytest.mark.skipif(not MODELO_INTENCOES.exists(), reason="rode python -m src.pln.intencoes")
@pytest.mark.parametrize("texto,intencao", [
    ("isso é reciclável?", "reciclavel"),
    ("qual a cor da lixeira", "cor_lixeira"),
    ("onde fica o ecoponto mais próximo", "onde_descartar"),
    ("oi", "saudacao"),
])
def test_intencoes(texto, intencao):
    from src.pln.intencoes import ClassificadorIntencoes
    assert ClassificadorIntencoes().prever(texto).intencao == intencao
