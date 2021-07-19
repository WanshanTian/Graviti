from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D
import os
import cv2
import csv

# configuration
root_path = "E:\\ShannonT\\dL-datasets\\wheat-detection"
dataset_name = "Global Wheat Detection"

# generate catalog
if "catalog.json" not in os.listdir(root_path):
    catalog(root_path, "wheat")

# create dataset
gas = GAS(access_key)
if dataset_name not in list(gas.list_dataset_names()):
    gas.create_dataset(dataset_name)

# load catalog
dataset = Dataset(dataset_name)
dataset.load_catalog(os.path.join(root_path, "catalog.json"))

# acquire bounding box
imgs_lable_dict = {}
with open(os.path.join(root_path, "train.csv"), "r") as file:
    csv_file = csv.reader(file)
    csv_file.__next__()
    for row in csv_file:
        if row[0] + ".jpg" not in imgs_lable_dict.keys():
            imgs_lable_dict[row[0] + ".jpg"] = []
        imgs_lable_dict[row[0] + ".jpg"].append([float(i) for i in row[3].split("[")[1].split("]")[0].split(",")])

# create dataloader
for k in ["train", "test"]:
    segment = dataset.create_segment(k)
    imgs_name = os.listdir(os.path.join(root_path, k))
    image_files_path = [os.path.join(os.path.join(root_path, k), i) for i in imgs_name]

    # 读取标签
    for i in range(len(image_files_path)):
        data = Data(image_files_path[i])
        if imgs_name[i] in imgs_lable_dict.keys():
            labels = imgs_lable_dict[imgs_name[i]]
        else:
            labels = []
        if len(labels) != 0:
            data.label.box2d = []
            for label in labels:
                cx = int(label[0])
                cy = int(label[1])
                w = int(label[2])
                h = int(label[3])
                data.label.box2d.append(LabeledBox2D.from_xywh(cx, cy, w, h,
                                                               category="wheat",
                                                               # attributes={"occluded": box["occluded"]}))
                                                               ))
        segment.append(data)
dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")
