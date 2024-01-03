import math
import os
import random
import shutil
from pathlib import Path

import click


@click.command()
@click.option("--training_mode", default="offline", show_default=True,
    type=click.Choice(["offline", "online"]), show_choices=True,
    help="Usage mode can be either 'webcam' or 'video'")
def split_data(training_mode: str):
    split_ratio: dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1}
    classes: list[str] = ["fake", "real"]

    dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))

    output_folder_path = Path.joinpath(Path.joinpath(dir_path, "Datasets"), "SplitData")
    input_folder_path = Path.joinpath(Path.joinpath(dir_path, "Datasets"), "all")

    training_mode: str = "offline"

    try:
        shutil.rmtree(output_folder_path)
    except OSError as error:
        os.mkdir(output_folder_path)

    # ----creating directories----
    os.makedirs(Path.joinpath(Path.joinpath(output_folder_path, "train"), "images"), exist_ok=True)
    os.makedirs(Path.joinpath(Path.joinpath(output_folder_path, "train"), "labels"), exist_ok=True)
    os.makedirs(Path.joinpath(Path.joinpath(output_folder_path, "val"), "images"), exist_ok=True)
    os.makedirs(Path.joinpath(Path.joinpath(output_folder_path, "val"), "labels"), exist_ok=True)
    os.makedirs(Path.joinpath(Path.joinpath(output_folder_path, "test"), "images"), exist_ok=True)
    os.makedirs(Path.joinpath(Path.joinpath(output_folder_path, "test"), "labels"), exist_ok=True)

    # ----getting the names----
    list_names: list[str] = os.listdir(input_folder_path)
    unique_names: list[str] = []

    for name in list_names:
        unique_names.append(name.split(".")[0])

    unique_names = list(set(unique_names))

    random.shuffle(unique_names)

    len_data: int = len(unique_names)
    len_train: int = math.ceil(len_data*split_ratio["train"])
    len_val: int = math.ceil(len_data*split_ratio["val"])
    len_test: int = len_data - len_train - len_val

    data_type2unique_names: dict[str, list[str]] = dict()
    data_type2unique_names["train"] = unique_names[:len_train]
    data_type2unique_names["val"] = unique_names[len_train: len_train + len_val]
    data_type2unique_names["test"] = unique_names[len_train + len_val:]

    data_types: list[str] = ["train", "val", "test"]
    for data_type in data_types:
        for filename in data_type2unique_names[data_type]:
            shutil.copy(Path.joinpath(input_folder_path, f"{filename}.jpg"),
                Path.joinpath(Path.joinpath(Path.joinpath(output_folder_path,
                    data_type), "images"), f"{filename}.jpg"))

            shutil.copy(Path.joinpath(input_folder_path, f"{filename}.txt"),
                Path.joinpath(Path.joinpath(Path.joinpath(output_folder_path,
                    data_type), "labels"), f"{filename}.txt"))

    # ----creating dataYaml file----
    if training_mode == "online":
        data_yaml: str = f"path: ../Data\n" + \
            "train: ../train/images\n" + \
            "val: ../val/images\n" + \
            "test: ../test/images\n" + \
            "\n" + \
            f"nc: {len(classes)}\n" + \
            f"names: {classes}\n"
    elif training_mode == "offline":
        data_yaml: str = f"path: {output_folder_path}\n" + \
            "train: train/images\n" + \
            "val: val/images\n" + \
            "test: test/images\n" + \
            "\n" + \
            f"nc: {len(classes)}\n" + \
            f"names: {classes}\n"

    with open(Path.joinpath(output_folder_path, "data.yaml"), "w") as output:
        output.write(data_yaml)

if __name__ == "__main__":
    split_data()
    