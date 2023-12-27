import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import math
from pathlib import Path

from time import time


import os

file_path = Path(os.path.basename(__file__))
print(file_path)
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
print(dir_path)

result = Path.joinpath(dir_path, file_path)
print(result)

outputFolderPath = Path.joinpath(result.parents[0], "Datasets/DataCollected")
print(outputFolderPath)

classID = 1  # 0 means fake and 1 means real

debug = False
save = True

offsetPercentageW = 10
offsetPercentageH = 20

confidence = 0.8

camWidth, camHeight = 640, 480

floatingPoints = 6

blurThreshold = 60  # larger means more focus

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()
while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # true if face is blur else false
    listInfo = []  # normalized values and a class name for the label text file

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        # center = bboxs[0]["center"]
        # cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # ----check the score----
            if score > confidence:

                # ----adding offset to the detected face----
                offsetW = (offsetPercentageW/100)*w

                x = x - math.ceil(offsetW)
                w = w + math.ceil(offsetW*2)

                offsetH = (offsetPercentageH/100)*h

                y = y - math.ceil(offsetH*3)
                h = h + math.ceil(offsetH*3.5)

                # ----to avoid values below 0----
                x = max(x, 0)
                y = max(y, 0)
                w = max(w, 0)
                h = max(h, 0)

                # ----find blurriness----

                imageFace = img[y:y + h, x: x + w]
                # cv2.imshow("face", imageFace)
                blurValue = math.ceil(cv2.Laplacian(imageFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ----normalize values----

                print(img.shape)
                imageHeight, imageWidth, _ = img.shape

                xNormalized = x / imageWidth
                xCenter, yCenter = x + w / 2, y + h/2

                xCenterNormalized = round(xCenter/imageWidth, floatingPoints)
                yCenterNormalized = round(yCenter/imageHeight, floatingPoints)

                wNormalized = round(w/imageWidth, floatingPoints)
                hNormalized = round(h/imageHeight, floatingPoints)

                # ----to avoid values above 1----
                xCenterNormalized = min(xCenterNormalized, 1)
                yCenterNormalized = min(yCenterNormalized, 1)
                wNormalized = min(wNormalized, 1)
                hNormalized = min(hNormalized, 1)

                # YOLO requires this format
                listInfo.append(f"{classID} {xCenterNormalized} {yCenterNormalized} {wNormalized} {hNormalized}\n")

                # ----drawing----

                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f"Score: {math.floor(score*100)}%; Blur: {blurValue}", (x, y-20),
                                   scale=1.5, thickness=3)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f"Score: {math.floor(score * 100)}%; Blur: {blurValue}", (x, y - 20),
                                       scale=1.5, thickness=3)

        # ---- to save ----
        if save:
            if all(listBlur) and listBlur != []:  # all the faces aren't blurred
                # ---- save image ====
                timeNow = "".join(str(time()).split("."))
                # print(timeNow)
                print(f"{outputFolderPath}/{timeNow}.jpg")
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

                # ----save label text file----
                with open(f"{outputFolderPath}/{timeNow}.txt", "a") as f:
                    for info in listInfo:
                        f.write(info)

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
