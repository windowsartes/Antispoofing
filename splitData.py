import math
import os
import random
import shutil
from itertools import islice

from pathlib import Path

splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

file_path = Path(os.path.basename(__file__))
dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

result = Path.joinpath(dir_path, file_path)

outputFolderPath = Path.joinpath(result.parents[0], "Datasets/SplitData")
inputFolderPath = Path.joinpath(result.parents[0], "Datasets/all")

trainingMode = "offline"

try:
    shutil.rmtree(outputFolderPath)
    print("Removed")
except OSError as error:
    os.mkdir(outputFolderPath)


# ----creating directories-----
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)


# print(inputFolderPath)
# ----getting the names----
listNames = os.listdir(inputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))

# print(uniqueNames)

# ----shuffle----

random.shuffle(uniqueNames)

lenData = len(uniqueNames)
lenTrain = math.ceil(lenData*splitRatio["train"])
lenVal = math.ceil(lenData*splitRatio["val"])
lenTest = lenData - lenTrain - lenVal

dataType2UniqueNames = dict()
dataType2UniqueNames["train"] = uniqueNames[:lenTrain]
dataType2UniqueNames["val"] = uniqueNames[lenTrain: lenTrain + lenVal]
dataType2UniqueNames["test"] = uniqueNames[lenTrain + lenVal:]

dataTypes = ["train", "val", "test"]
for dataType in dataTypes:
    for filename in dataType2UniqueNames[dataType]:
        shutil.copy(f"{inputFolderPath}/{filename}.jpg", f"{outputFolderPath}/{dataType}/images/{filename}.jpg")
        shutil.copy(f"{inputFolderPath}/{filename}.txt", f"{outputFolderPath}/{dataType}/labels/{filename}.txt")

# ----creating dataYaml file----

if trainingMode == "online":
    dataYaml = f"path: ../Data\ntrain: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}\n"
elif trainingMode == "offline":
    dataYaml = f"path: {outputFolderPath}\n\
train: train/images\n\
val: val/images\n\
test: test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}\n"

with open(f"{outputFolderPath}/data.yaml", "w") as f:
    f.write(dataYaml)
