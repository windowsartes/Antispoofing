import os
from pathlib import Path

from ultralytics import YOLO


file_path: os.PathLike = Path(os.path.basename(__file__))
dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))
file_path = Path.joinpath(dir_path, file_path)

dataYamlPath: os.PathLike = Path.joinpath(file_path.parents[0], "Datasets/SplitData/data.yaml")

def train(model: YOLO, num_epochs: int = 10):
    model.train(data=dataYamlPath, epochs=num_epochs)


if __name__ == "__main__":
    model: YOLO = YOLO("yolov8n.pt")

    train(model, 10)
