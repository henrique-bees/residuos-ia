"""Inferência: recebe uma imagem e devolve as probabilidades por classe."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from src.config import CLASSES, MODELO_CNN
from src.visao_computacional.modelo import TRANSF_AVAL, criar_modelo


@dataclass
class ResultadoVisao:
    classe: str                       # nome da pasta do TrashNet (ex.: "glass")
    confianca: float                  # probabilidade da classe prevista
    probabilidades: dict[str, float]  # todas as classes, ordenadas da maior p/ menor

    @property
    def margem(self) -> float:
        """Diferença entre a 1ª e a 2ª classe mais prováveis."""
        p = list(self.probabilidades.values())
        return p[0] - p[1] if len(p) > 1 else p[0]


class ClassificadorResiduos:
    def __init__(self, caminho=MODELO_CNN, disp: str | None = None):
        self.disp = disp or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = criar_modelo(pre_treinado=False)
        self.modelo.load_state_dict(torch.load(caminho, map_location=self.disp))
        self.modelo.to(self.disp).eval()

    @torch.no_grad()
    def classificar(self, imagem: Image.Image) -> ResultadoVisao:
        x = TRANSF_AVAL(imagem.convert("RGB")).unsqueeze(0).to(self.disp)
        probs = torch.softmax(self.modelo(x), dim=1)[0].cpu().tolist()
        ordenado = dict(sorted(zip(CLASSES, probs), key=lambda kv: kv[1], reverse=True))
        classe = next(iter(ordenado))
        return ResultadoVisao(classe, ordenado[classe], ordenado)
