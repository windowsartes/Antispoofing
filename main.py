import math
import os
from pathlib import Path

import cv2
import cvzone
from ultralytics import YOLO


confidence_threshold: float = 0.8
mode: str = "webcam"

capture: cv2.VideoCapture | None = None

if mode == "webcam":
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
else:
    capture = cv2.VideoCapture("*path_to_video*")

dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))

path_to_weights: os.PathLike = Path.joinpath(Path.joinpath(dir_path, "saved_weights"), "best.pt")

model: YOLO = YOLO(path_to_weights)

class_names: list[str] = ["fake", "real"]

while True:
    success, image = capture.read()
    results = model(image, stream=True, verbose=False)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1: int | float = 0
            y1: int | float = 0
            x2: int | float = 0
            y2: int | float = 0

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w: int = 0
            h: int = 0
            w, h = x2 - x1, y2 - y1

            current_confidence: float = box.conf[0]
            current_class: int = int(box.cls[0])

            if current_confidence > confidence_threshold:
                if class_names[current_class] == "real":
                    color: tuple[int, int, int] = (0, 255, 0)
                else:
                    color: tuple[int, int, int] = (0, 0, 255)

                cvzone.cornerRect(image, (x1, y1, w, h), colorC = color, colorR = color)
                cvzone.putTextRect(image, f"{class_names[current_class].upper()} {round(current_confidence.item(), 2) * 100}%",
                                   (max(0, x1), max(35, y1)), scale = 2, thickness = 4,
                                   colorR = color, colorB = color)

    cv2.imshow("Image", image)
    cv2.waitKey(1)
