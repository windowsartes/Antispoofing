import os
from pathlib import Path

from ultralytics import YOLO


dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))

data_yaml_path: os.PathLike = Path.joinpath(Path.joinpath(Path.joinpath(
    Path.joinpath(dir_path, "Datasets"), "SplitData"), "data.yaml"))

def train(model: YOLO, num_epochs: int = 10):
    """Trains given model on given number of epochs

    :param model: model you want to train
    :type model: YOLO
    :param num_epochs: number of epochs during which you want to train your model, defaults to 10
    :type num_epochs: int, optional
    """
    model.train(data=data_yaml_path, epochs=num_epochs)


if __name__ == "__main__":
    model: YOLO = YOLO("yolov8n.pt")

    train(model, 20)
