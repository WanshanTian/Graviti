from common.dataset_initial import initial
from tensorbay.label import Classification
from tensorbay.dataset import Data
import os
from tensorbay.client import config


config.timeout = 40
config.max_retries = 4

root_path = "E:\\ShannonT\\dL-datasets\\logo-images-dataset-master"
dataset_name = "logos-100"

contents = os.listdir(root_path)
if "logo_list.txt" in contents:
    contents.remove("logo_list.txt")

if "catalog.json" in contents:
    contents.remove("catalog.json")

classes_path = os.path.join(root_path, "logo_list.txt")
classes = []
with open(classes_path, "r") as f:
    for line in f.readlines():
        l = " ".join(line.split("\t"))
        classes.append(l.strip("\n"))

initial = initial(root_path, dataset_name, ["CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

for i in contents:
    sub = os.listdir(os.path.join(root_path, i))
    for j in sub:
        segment = dataset.create_segment(j)
        imgs = os.listdir(os.path.join(os.path.join(root_path, i), j))
        for iii in imgs:
            if iii.split(".")[-1] != "png":
                imgs.remove(iii)
        for img in imgs:
            img_path = os.path.join(os.path.join(os.path.join(root_path, i), j), img)
            data = Data(img_path, target_remote_path=j + img)
            data.label.classification = Classification(j)
            segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
