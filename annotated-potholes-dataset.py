from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D, Classification
import os
import json
from xml.dom.minidom import parse
import xml.dom.minidom

# configuration
root_path = "G:\\shannon\\deeplearning_dataset\\annotated-potholes-dataset"
dataset_name = "annotated-potholes-dataset"

# generate catalog
if "catalog.json" not in os.listdir(root_path):
    catalog("box2d", root_path, "pothole")

# create dataset
gas = GAS(access_key)
if dataset_name not in list(gas.list_dataset_names()):
    gas.create_dataset(dataset_name)

# load catalog
dataset = Dataset(dataset_name)
dataset.load_catalog(os.path.join(root_path, "catalog.json"))

with open(os.path.join(root_path, "splits.json"), encoding='utf-8') as f:
    line = f.readline()
    all = json.loads(line)

for k in list(all.keys()):
    segment = dataset.create_segment(k)
    for label_file in all[k]:
        img_path = os.path.join(os.path.join(root_path, "annotated-images"), label_file.split(".")[0]+".jpg")
        image_label_path = os.path.join(os.path.join(root_path, "annotated-images"), label_file)

        DOMTree = xml.dom.minidom.parse(image_label_path)
        collection = DOMTree.documentElement
        boundingbox = collection.getElementsByTagName("object")

        labels = []
        for i in boundingbox:
            category = i.getElementsByTagName("name")[0].childNodes[0].data
            tmp = []
            tmp.append(float(
                [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")][
                    0]))
            tmp.append(float(
                [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")][
                    0]))
            tmp.append(float(
                [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")][
                    0]))
            tmp.append(float(
                [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")][
                    0]))
            tmp.append(category)
            labels.append(tmp)
        data = Data(img_path)

        if len(labels) != 0:
            data.label.box2d = []
            for i in range(len(labels)):
                xmin = labels[i][0]
                ymin = labels[i][1]
                xmax = labels[i][2]
                ymax = labels[i][3]
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category="pothole",
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        segment.append(data)
dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")
