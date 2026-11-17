"""Obtenção, divisão estratificada e carregamento do dataset TrashNet.

Uso:
    python -m src.visao_computacional.dataset --baixar   # baixa e extrai o zip
    python -m src.visao_computacional.dataset --dividir  # gera dados/splits/*.csv
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import CLASSES, DADOS, DIR_IMAGENS, DIR_SPLITS, SEMENTE

URL_TRASHNET = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
EXTENSOES = {".jpg", ".jpeg", ".png"}


def baixar(url: str = URL_TRASHNET) -> None:
    """Baixa o zip do TrashNet e extrai em dados/trashnet/<classe>/."""
    destino_zip = DADOS / "dataset-resized.zip"
    if not destino_zip.exists():
        print(f"Baixando {url} ...")
        urllib.request.urlretrieve(url, destino_zip)
    tmp = DADOS / "_tmp_trashnet"
    with zipfile.ZipFile(destino_zip) as z:
        z.extractall(tmp)
    # O zip contém dataset-resized/<classe>/...
    origem = next(p for p in tmp.rglob("cardboard") if p.is_dir()).parent
    DIR_IMAGENS.mkdir(parents=True, exist_ok=True)
    for classe in CLASSES:
        alvo = DIR_IMAGENS / classe
        if alvo.exists():
            shutil.rmtree(alvo)
        shutil.move(str(origem / classe), alvo)
    shutil.rmtree(tmp)
    print(f"Imagens extraídas em {DIR_IMAGENS}")


def listar_imagens(raiz: Path = DIR_IMAGENS) -> pd.DataFrame:
    """Lista as imagens válidas. Arquivos corrompidos são descartados (limpeza)."""
    linhas, descartadas = [], 0
    for classe in CLASSES:
        for arq in sorted((raiz / classe).glob("*")):
            if arq.suffix.lower() not in EXTENSOES:
                continue
            try:
                with Image.open(arq) as im:
                    im.verify()
            except Exception:
                descartadas += 1
                continue
            linhas.append({"caminho": str(arq.relative_to(raiz)), "classe": classe})
    if descartadas:
        print(f"Aviso: {descartadas} imagens corrompidas descartadas")
    return pd.DataFrame(linhas)


def dividir(prop_val: float = 0.15, prop_teste: float = 0.15) -> dict[str, pd.DataFrame]:
    """Divisão estratificada treino/validação/teste (70/15/15 por padrão)."""
    df = listar_imagens()
    if df.empty:
        raise FileNotFoundError(f"Nenhuma imagem em {DIR_IMAGENS}. Rode com --baixar antes.")
    treino, resto = train_test_split(
        df, test_size=prop_val + prop_teste, stratify=df["classe"], random_state=SEMENTE
    )
    val, teste = train_test_split(
        resto,
        test_size=prop_teste / (prop_val + prop_teste),
        stratify=resto["classe"],
        random_state=SEMENTE,
    )
    DIR_SPLITS.mkdir(parents=True, exist_ok=True)
    splits = {"treino": treino, "validacao": val, "teste": teste}
    for nome, parte in splits.items():
        parte.to_csv(DIR_SPLITS / f"{nome}.csv", index=False)
    resumo = pd.DataFrame({n: p["classe"].value_counts() for n, p in splits.items()})
    print(resumo.to_string())
    return splits


def carregar_split(nome: str) -> pd.DataFrame:
    return pd.read_csv(DIR_SPLITS / f"{nome}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baixar", action="store_true")
    ap.add_argument("--dividir", action="store_true")
    a = ap.parse_args()
    if a.baixar:
        baixar()
    if a.dividir:
        dividir()
