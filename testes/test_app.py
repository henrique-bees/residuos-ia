"""Smoke test do app Streamlit: carrega sem exceções com os modelos treinados."""
import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODELO_CNN, MODELO_INTENCOES, RAIZ

pytestmark = pytest.mark.skipif(
    not (MODELO_CNN.exists() and MODELO_INTENCOES.exists()),
    reason="rode os treinos (README) para gerar os modelos antes deste teste",
)


def test_app_carrega_sem_erro():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert not at.error
    assert at.title[0].value == "♻️ Descarte Inteligente — Maceió"
