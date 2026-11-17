"""Avalia o modelo salvo em fotos reais de campo (fora do domínio do TrashNet).

Coloque as fotos em dados/teste_campo/<classe>/*.jpg (classe = cardboard, glass, metal,
paper, plastic ou trash) e rode:

    python -m src.visao_computacional.avaliar_campo

Gera resultados/metricas_cnn_campo.json (acurácia, F1-macro, por classe, taxa de recusa
do agente e acurácia entre as respostas aceitas) e o gráfico de confusão correspondente.
"""
from __future__ import annotations

from PIL import Image

from src.config import CLASSES, CLASSES_PT, DADOS, LIMIAR_CONFIANCA
from src.machine_learning.metricas import relatorio
from src.visao_computacional.classificador import ClassificadorResiduos

DIR_TESTE_CAMPO = DADOS / "teste_campo"
EXTENSOES = {".jpg", ".jpeg", ".png"}


def avaliar() -> dict | None:
    imagens = []
    for classe in CLASSES:
        pasta = DIR_TESTE_CAMPO / classe
        if not pasta.exists():
            continue
        for arq in sorted(pasta.glob("*")):
            if arq.suffix.lower() in EXTENSOES:
                imagens.append((arq, classe))

    if not imagens:
        print(f"Nenhuma foto encontrada em {DIR_TESTE_CAMPO}/<classe>/.")
        print("Crie as pastas por classe (cardboard, glass, metal, paper, plastic, trash), "
              "coloque ~20 fotos reais por classe e rode este script de novo.")
        return None

    clf = ClassificadorResiduos()
    y_true, y_pred, confiancas = [], [], []
    for caminho, classe in imagens:
        r = clf.classificar(Image.open(caminho))
        y_true.append(CLASSES_PT[classe])
        y_pred.append(CLASSES_PT[r.classe])
        confiancas.append(r.confianca)

    resumo = relatorio(y_true, y_pred, [CLASSES_PT[c] for c in CLASSES], "cnn_campo")

    aceitas = [c >= LIMIAR_CONFIANCA for c in confiancas]
    resumo["n_imagens"] = len(imagens)
    resumo["taxa_recusa"] = 1 - sum(aceitas) / len(aceitas)
    if any(aceitas):
        acertos_aceitas = sum(
            1 for ok, t, p in zip(aceitas, y_true, y_pred) if ok and t == p
        )
        resumo["acuracia_entre_aceitas"] = acertos_aceitas / sum(aceitas)
    else:
        resumo["acuracia_entre_aceitas"] = None

    print(f"[cnn_campo] {len(imagens)} fotos | taxa de recusa "
          f"(confiança < {LIMIAR_CONFIANCA:.0%}) = {resumo['taxa_recusa']:.1%} | "
          f"acurácia entre aceitas = {resumo['acuracia_entre_aceitas']}")

    import json
    from src.config import RESULTADOS
    (RESULTADOS / "metricas_cnn_campo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resumo


if __name__ == "__main__":
    avaliar()
