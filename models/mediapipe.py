from typing import Tuple

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_finger_on_frame(img, hands_detector) -> Tuple[float, float]:
    """
    Функция определяет точку на указательном пальце через обнаружение ключевых точек кисти с помощью mediapipe
    :param img: np.ndarray
    :param hands_detector: a hand detector
    :return: x, y
    """

    hand_landmarks = hands_detector.process(img)

    x, y = 0, 0
    if hand_landmarks.multi_hand_landmarks:
        for h_landmark in hand_landmarks.multi_hand_landmarks:
            point_finger = h_landmark.landmark[8]
            x, y = point_finger.x, point_finger.y

    return x, y
