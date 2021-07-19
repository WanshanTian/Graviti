from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D, Classification
import os

# configuration
root_path = "G:\\shannon\\deeplearning_dataset\\1-ExDark"
dataset_name = "ExDark"

# generate catalog
if "catalog.json" not in os.listdir(root_path):
    catalog("classification", root_path, "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog",
            "Motorbike", "people",
            "Table")

# create dataset
gas = GAS(access_key)
if dataset_name not in list(gas.list_dataset_names()):
    gas.create_dataset(dataset_name)

# load catalog
dataset = Dataset(dataset_name)
dataset.load_catalog(os.path.join(root_path, "catalog.json"))

segment = dataset.create_segment("Train & Test")
classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "people",
           "Table"]

for type in classes:
    imgs_name = os.listdir(os.path.join(root_path, type))
    imgs_path = [os.path.join(os.path.join(root_path, type), i) for i in imgs_name]
    for img in imgs_path:
        data = Data(img)
        data.label.classification = Classification(type)
        segment.append(data)

dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")
