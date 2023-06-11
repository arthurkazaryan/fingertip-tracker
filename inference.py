import os
from argparse import ArgumentParser
from collections import deque

import cv2
import numpy as np

from models.mediapipe import mp_hands, detect_hand_on_frame, detect_finger_on_frame
from models.mobilenet import load_model
from models.mobilenet import detect_finger_on_frame as detect_finger_on_frame_mobilenet
from utils import cap_from_video, cap_from_webcam, get_meta_from_cap

coords = deque(maxlen=25)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="output.mp4", help="Path to an output file", required=False)
    subparsers = parser.add_subparsers(help="sub-command help")

    camera_parser = subparsers.add_parser("camera", help="Inference from a web camera")
    camera_parser.set_defaults(mode="camera")
    camera_parser.add_argument("-m", "--model", type=str, default="mediapipe", help="Model to use in inference", required=True)
    camera_parser.add_argument("-p", "--path", type=str, required=False, help="Path to the model weights", default="models/weights/model_2.pth")

    video_parser = subparsers.add_parser("video", help="Inference from video")
    video_parser.set_defaults(mode="video")
    video_parser.add_argument("-v", "--video", type=str, required=True)
    video_parser.add_argument("-p", "--path", type=str, required=False, default="models/weights/model_2.pth")
    video_parser.add_argument("-m", "--model", type=str, default="mediapipe", help="Model to use in inference", required=True)

    args = parser.parse_args()

    if args.mode == "camera":
        capture = cap_from_webcam()
    else:
        assert os.path.isfile(args.video), "Cannot find video file"
        capture = cap_from_video(args.video)
    mobilenet = None
    if args.model == "mobilenet":
        assert os.path.isfile(args.path), f"Cannot find weights on {args.path}"
        mobilenet = load_model(args.path)
    width, height, frames, fps = get_meta_from_cap(capture)

    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # video_path.suffix
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
    ) as hands:
        while capture.isOpened():
            ret, frame = capture.read()
            x1, y1, x2, y2 = detect_hand_on_frame(img=frame[:, :, ::-1], hands_detector=hands)
            if x1 and y1:
                if args.model == "mobilenet":
                    img_crop = frame[:, :, ::-1][y1:y2, x1:x2]
                    x, y = detect_finger_on_frame_mobilenet(img_crop, mobilenet)
                    if x and y:
                        x += x1
                        y += y1
                elif args.model == "mediapipe":
                    x, y = detect_finger_on_frame(img=frame[:, :, ::-1], hands_detector=hands)
                    x, y = [int(x * width), int(y * height)]
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame = cv2.polylines(frame, [np.array(coords)], False, (255, 0, 0), 2)
                coords.append([x, y])
            cv2.imshow('Draw with your point finger ', frame)
            if video_writer:
                video_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_writer.release()
                break
