from pathlib import Path
from typing import Tuple

import cv2
from torch.cuda import is_available

device = 'cuda' if is_available() else 'cpu'

def cap_from_video(file_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(file_path))
    return cap


def cap_from_webcam(cam_id: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_id)
    return cap


def get_meta_from_cap(cap: cv2.VideoCapture) -> Tuple[int, int, int, int]:
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return width, height, frames, fps
