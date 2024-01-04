import os
from pathlib import Path

import click
from ultralytics import YOLO


dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))

data_yaml_path: os.PathLike = Path.joinpath(Path.joinpath(Path.joinpath(
    Path.joinpath(dir_path, "Datasets"), "SplitData"), "data.yaml"))

@click.command()
@click.option("--n_epochs", default=20, show_default=True, type=int,  help="Number of epochs.")
def train(n_epochs):
    """Trains given model during given number of epochs

    :param num_epochs: number of epochs during which you want to train your model, defaults to 20
    :type num_epochs: int, optional
    """
    model: YOLO = YOLO("yolov8n.pt")
    model.train(data=data_yaml_path, epochs=num_epochs)


if __name__ == "__main__":
    train()
