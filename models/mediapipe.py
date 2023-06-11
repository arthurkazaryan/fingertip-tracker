from typing import Tuple

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_hand_on_frame(img, hands_detector):
    """
    Функция определяет ббоксы руки при помощи определения точек через mediapipe
    :param img: np.ndarray
    :param hands_detector: a hand detector
    :return: x1, y1, x2, y2
    """
    height, width, *_ = img.shape
    hand_landmarks = hands_detector.process(img)
    x1, x2 = None, None
    y1, y2 = None, None
    if hand_landmarks.multi_hand_landmarks:
        landmark = hand_landmarks.multi_hand_landmarks[0]
        x_points = [point.x for point in landmark.landmark]
        y_points = [point.y for point in landmark.landmark]
        min_x, max_x = min(x_points), max(x_points)
        min_y, max_y = min(y_points), max(y_points)
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        x1, x2 = min_x - delta_x * 0.0, max_x + delta_x * 0.0
        y1, y2 = min_y - delta_y * 0.0, max_y + delta_y * 0.0
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > 1: x2 = 1
        if y2 > 1: y2 = 1
        x1, y1, x2, y2 = int(width * x1), int(height * y1), int(width * x2), int(height * y2)

    return x1, y1, x2, y2



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
