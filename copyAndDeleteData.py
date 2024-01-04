import os
import shutil
from pathlib import Path


def copy_and_delete(inputFolderPath: os.PathLike, outputFolderPath: os.PathLike) -> None:
    """ Copy all files from inputFolderPath to outputFolderParh

    :param inputFolderPath: the folder from which you want to copy and delete files;
    :type inputFolderPath: os.PathLike
    :param outputFolderPath: the folder where you want to copy files;
    :type outputFolderPath: os.PathLike
    """
    for file in os.listdir(inputFolderPath):
        shutil.copy(Path.joinpath(inputFolderPath, file), Path.joinpath(outputFolderPath, file))
        os.remove(Path.joinpath(inputFolderPath, file))


if __name__ == "__main__":
    dir_path: os.PathLike = Path(os.path.dirname(os.path.realpath(__file__)))

    inputFolderPath: os.PathLike = Path.joinpath(Path.joinpath(dir_path, "Datasets"), "DataCollected")
    outputFolderPath: os.PathLike = Path.joinpath(Path.joinpath(dir_path, "Datasets"), "all")

    copy_and_delete(inputFolderPath, outputFolderPath)
