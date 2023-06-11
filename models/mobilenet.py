from typing import Tuple

import albumentations as A
import timm
import torch.nn as nn
from torch import load
from albumentations.pytorch import ToTensorV2

from utils import device


preprocessing = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(),
        ToTensorV2()
    ])


def apply_preprocessing(img):
    return preprocessing(image=img)['image']


def load_model(model_path: str):
    model = timm.create_model('mobilenetv2_050', pretrained=True, num_classes=0)
    model.global_pool = nn.Flatten()
    model.classifier = nn.Sequential(
        nn.Linear(62720, 3),
        nn.Sigmoid()
    )
    model.load_state_dict(load(model_path))
    model.eval()
    model.to(device)
    return model


def detect_finger_on_frame(img, hands_detector) -> Tuple[int, int]:

    height, width, *_ = img.shape
    img = apply_preprocessing(img)

    pred = hands_detector(img.unsqueeze(0).to(device))
    x, y, *_ = pred.cpu().detach().numpy()[0]

    x, y = int(x*width), int(y*height)

    return x, y
