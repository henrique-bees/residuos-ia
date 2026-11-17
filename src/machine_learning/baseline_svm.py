"""Baseline clássico de ML: características manuais (HOG + histograma de cor) + SVM.

Serve de comparação com a CNN no artigo (ML clássico x deep learning).

Uso:
    python -m src.machine_learning.baseline_svm
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CLASSES, CLASSES_PT, DIR_IMAGENS, MODELO_BASELINE, SEMENTE
from src.machine_learning.metricas import relatorio
from src.visao_computacional.dataset import carregar_split

TAM = (128, 128)


def extrair_caracteristicas(img: Image.Image) -> np.ndarray:
    """Vetor = HOG (forma/bordas) em tons de cinza + histograma HSV (cor)."""
    img = img.convert("RGB").resize(TAM)
    cinza = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    v_hog = hog(cinza, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    hsv = np.asarray(img.convert("HSV"))
    v_cor = np.concatenate([
        np.histogram(hsv[..., c], bins=16, range=(0, 255), density=True)[0] for c in range(3)
    ])
    return np.concatenate([v_hog, v_cor]).astype(np.float32)


def _matriz(df):
    X = np.stack([extrair_caracteristicas(Image.open(DIR_IMAGENS / c)) for c in df["caminho"]])
    y = np.array([CLASSES.index(c) for c in df["classe"]])
    return X, y


def treinar(amostra: int = 0):
    tr, va, te = (carregar_split(n) for n in ("treino", "validacao", "teste"))
    # Treino + validação juntos: o GridSearch já faz validação cruzada interna
    tr = pd.concat([tr, va])
    if amostra:
        tr, te = (d.groupby("classe").head(amostra) for d in (tr, te))
    X, y = _matriz(tr)
    pipe = make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMENTE))
    busca = GridSearchCV(pipe, {"svc__C": [1, 10], "svc__gamma": ["scale", 1e-3]},
                         cv=3, n_jobs=-1, scoring="f1_macro")
    busca.fit(X, y)
    print("Melhores hiperparâmetros:", busca.best_params_)
    MODELO_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(busca.best_estimator_, MODELO_BASELINE)

    Xte, yte = _matriz(te)
    return relatorio(yte.tolist(), busca.predict(Xte).tolist(),
                     [CLASSES_PT[c] for c in CLASSES], "baseline_svm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra", type=int, default=0)
    treinar(ap.parse_args().amostra)
