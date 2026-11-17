"""Reconhecimento de intenções: TF-IDF (palavras + n-gramas de caracteres) + classificador.

Compara três modelos (Regressão Logística, Naive Bayes, SVM linear) com validação
cruzada estratificada e avalia o melhor em um conjunto de teste separado.

Uso:
    python -m src.pln.intencoes
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from src.config import CSV_INTENCOES, LIMIAR_INTENCAO, MODELO_INTENCOES, RESULTADOS, SEMENTE
from src.machine_learning.metricas import relatorio
from src.pln.preprocessamento import normalizar


def _vetorizador():
    # n-gramas de caracteres deixam o modelo robusto a erros de digitação ("reciclavel", "resiclável")
    return FeatureUnion([
        ("palavras", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                       ngram_range=(2, 5), sublinear_tf=True)),
    ])


MODELOS = {
    "regressao_logistica": lambda: LogisticRegression(C=10, max_iter=2000),
    "naive_bayes": lambda: ComplementNB(alpha=0.3),
    "svm_linear": lambda: CalibratedClassifierCV(LinearSVC(C=1), cv=3),
}


def treinar_e_avaliar() -> dict:
    df = pd.read_csv(CSV_INTENCOES)
    treino, teste = train_test_split(df, test_size=0.25, stratify=df["intencao"],
                                     random_state=SEMENTE)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)

    comparacao = {}
    for nome, fabrica in MODELOS.items():
        pipe = make_pipeline(_vetorizador(), fabrica())
        notas = cross_val_score(pipe, treino["texto"], treino["intencao"], cv=cv, scoring="f1_macro")
        comparacao[nome] = {"f1_macro_cv_media": notas.mean(), "f1_macro_cv_desvio": notas.std()}
        print(f"{nome:22s} F1-macro (CV 5 folds) = {notas.mean():.3f} ± {notas.std():.3f}")

    melhor = max(comparacao, key=lambda n: comparacao[n]["f1_macro_cv_media"])
    print(f"Melhor modelo: {melhor}")
    pipe = make_pipeline(_vetorizador(), MODELOS[melhor]())
    pipe.fit(treino["texto"], treino["intencao"])

    nomes = sorted(df["intencao"].unique())
    resumo = relatorio(teste["intencao"].tolist(), pipe.predict(teste["texto"]).tolist(),
                       nomes, "intencoes")
    resumo["comparacao_cv"] = comparacao
    resumo["modelo_escolhido"] = melhor
    (RESULTADOS / "metricas_intencoes.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # Modelo final: re-treinado com todos os dados
    pipe.fit(df["texto"], df["intencao"])
    MODELO_INTENCOES.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELO_INTENCOES)
    return resumo


@dataclass
class ResultadoIntencao:
    intencao: str
    confianca: float
    probabilidades: dict[str, float]


class ClassificadorIntencoes:
    def __init__(self, caminho=MODELO_INTENCOES):
        self.pipe = joblib.load(caminho)

    def prever(self, texto: str) -> ResultadoIntencao:
        probs = self.pipe.predict_proba([texto])[0]
        ordenado = dict(sorted(zip(self.pipe.classes_, probs), key=lambda kv: kv[1], reverse=True))
        intencao, conf = next(iter(ordenado.items()))
        if conf < LIMIAR_INTENCAO:
            intencao = "fora_escopo"
        return ResultadoIntencao(intencao, float(conf), {k: float(v) for k, v in ordenado.items()})


if __name__ == "__main__":
    treinar_e_avaliar()
