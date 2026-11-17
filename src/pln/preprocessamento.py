"""Normalização de texto em português usada por todo o módulo de PLN."""
import re
import unicodedata


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos, sem pontuação, espaços únicos.

    Também expande abreviações comuns de chat ("vc", "pq", "pra"...).
    """
    t = remover_acentos(texto.lower())
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    abrev = {"vc": "voce", "pq": "porque", "q": "que", "tb": "tambem", "td": "tudo",
             "msm": "mesmo", "oq": "o que", "cmo": "como", "ond": "onde"}
    t = " ".join(abrev.get(p, p) for p in t.split())
    return t
