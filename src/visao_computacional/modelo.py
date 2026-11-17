"""Modelo de Visão Computacional: MobileNetV2 com transfer learning."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from src.config import CLASSES, DIR_IMAGENS

TAM = 224
MEDIA = [0.485, 0.456, 0.406]   # normalização do ImageNet
DESVIO = [0.229, 0.224, 0.225]

TRANSF_TREINO = transforms.Compose([
    transforms.RandomResizedCrop(TAM, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])
TRANSF_AVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(TAM),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])


class DatasetResiduos(Dataset):
    def __init__(self, df: pd.DataFrame, transform, raiz=DIR_IMAGENS):
        self.caminhos = [raiz / c for c in df["caminho"]]
        self.rotulos = [CLASSES.index(c) for c in df["classe"]]
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = Image.open(self.caminhos[i]).convert("RGB")
        return self.transform(img), self.rotulos[i]


def criar_modelo(pre_treinado: bool = True, congelar_base: bool = True) -> nn.Module:
    """MobileNetV2 (ImageNet) com a última camada trocada para 6 classes.

    Com congelar_base=True apenas o classificador é treinado (feature extraction);
    depois as últimas camadas podem ser liberadas para fine-tuning.
    """
    pesos = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pre_treinado else None
    m = models.mobilenet_v2(weights=pesos)
    if congelar_base:
        for p in m.features.parameters():
            p.requires_grad = False
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, len(CLASSES)),
    )
    return m


def liberar_ultimas_camadas(m: nn.Module, n_blocos: int = 4) -> None:
    """Fine-tuning: descongela os últimos n blocos convolucionais."""
    for bloco in list(m.features.children())[-n_blocos:]:
        for p in bloco.parameters():
            p.requires_grad = True
