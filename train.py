from ultralytics import YOLO
from pathlib import Path

import os

file_path = Path(os.path.basename(__file__))
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
result = Path.joinpath(dir_path, file_path)

dataYamlPath = Path.joinpath(result.parents[0], "Datasets/SplitData/data.yaml")


model = YOLO("yolov8n.pt")


def main():
    model.train(data=dataYamlPath, epochs=3)


if __name__ == "__main__":
    main()
