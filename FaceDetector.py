import math
import os
from pathlib import Path
from time import time

import cvzone
import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector


debug: bool = False
save: bool = False

dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))
if debug:
    print(dir_path)

output_folder_path: os.PathLike = Path.joinpath(Path.joinpath(dir_path, "Datasets"), "DataCollected")
if debug:
    print(output_folder_path)

class_id: int = 0  # 0 means fake and 1 means real

offset_percentage_w: int = 10
offset_percentage_h: int = 20

confidence_threshold: float = 0.8

camera_width: int = 640
camera_height: int = 480

precision: int = 6

blur_threshold: int = 70  # larger means more focus

capture: cv2.VideoCapture = cv2.VideoCapture(0)
capture.set(3, camera_width)
capture.set(4, camera_height)

detector: FaceDetector = FaceDetector()

while True:
    success: bool = False
    image: np.ndarray = np.empty(0)
    success, image = capture.read()

    image_out: np.ndarray = image.copy()

    bboxes: list[dict[str]] = []
    image, bboxes = detector.findFaces(image, draw=False)

    list_blur: list[bool] = []  # true if face is blur else false
    list_info: list[str] = []  # normalized values and a class name for the label text file

    if bboxes:
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxes:
            x: int = 0
            y: int = 0
            w: int = 0
            h: int = 0
            x, y, w, h = bbox["bbox"]

            current_confidence: float = bbox["score"][0]

            # ----check the current confidence----
            if current_confidence > confidence_threshold:
                # ----adding offset to the detected face----
                offset_w: float = (offset_percentage_w/100)*w

                x -= math.ceil(offset_w)
                w += math.ceil(offset_w*2)

                offset_h: float = (offset_percentage_h/100)*h

                y -= math.ceil(offset_h*3)
                h += math.ceil(offset_h*3.5)

                # ----to avoid values below 0----
                x = max(x, 0)
                y = max(y, 0)
                w = max(w, 0)
                h = max(h, 0)

                # ----find blurriness----
                face_crop: np.ndarray = image[y:y + h, x: x + w]

                blur_value: int = math.ceil(cv2.Laplacian(face_crop, cv2.CV_64F).var())
                if blur_value > blur_threshold:
                    list_blur.append(True)
                else:
                    list_blur.append(False)

                # ----normalize values----
                image_height: int = 0
                image_width: int = 0
                image_height, image_width, _ = image.shape

                x_center: float = x + w/2
                y_center: float =  y + h/2

                x_center_normalized: float = round(x_center/image_width, precision)
                y_center_normalized: float = round(y_center/image_height, precision)

                w_normalized: float = round(w/image_width, precision)
                h_normalized: float = round(h/image_height, precision)

                # ----to avoid values above 1----
                x_center_normalized = min(x_center_normalized, 1)
                y_center_normalized = min(y_center_normalized, 1)
                w_normalized = min(w_normalized, 1)
                h_normalized = min(h_normalized, 1)

                # YOLO requires this format
                list_info.append(f"{class_id} {x_center_normalized}" +
                    f"{y_center_normalized} {w_normalized} {h_normalized}\n")

                # ----drawing----
                cv2.rectangle(image_out, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(image_out, f"Score: {math.floor(current_confidence*100)}%;" +
                    f"Blur: {blur_value}", (x, y-20), scale = 1.5, thickness = 3)
    
                if debug:
                    cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(image, f"Score: {math.floor(current_confidence * 100)}%;" +
                        f"Blur: {blur_value}", (x, y - 20), scale = 1.5, thickness = 3)

        # ---- to save ----
        if save:
            if list_blur != [] and all(list_blur):  # all the faces aren't blurred
                # ---- save image ----
                time_now = "".join(str(time()).split("."))

                if debug:
                    print(Path.joinpathPath(output_folder_path, f"{time_now}.jpg"))
                
                cv2.imwrite(Path.joinpathPath(output_folder_path, f"{time_now}.jpg"), image)

                # ----save label text file----
                with open(Path.joinpathPath(output_folder_path, f"{time_now}.jpg"), "a") as output:
                    for info in list_info:
                        output.write(info)

    cv2.imshow("Image", image_out)
    cv2.waitKey(1)
