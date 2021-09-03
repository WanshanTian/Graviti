from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data, Dataset
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
from tensorbay import GAS
from config import access_key
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom

config.timeout = 40
config.max_retries = 4

root_path = "G:\download_dataset\AU_AIR\\auair2019data"
dataset_name = "AU_AIR"

label_path = "G:\download_dataset\AU_AIR\\auair2019annotations\\annotations.json"
with open(label_path, encoding='utf-8') as f:
    line = f.readline()
    all = json.loads(line)
classes = all.get("categories")
splits = {}
for i,j in zip(range(8), classes):
    splits[i] = j
# print(json.dumps(all, sort_keys=True, indent=4, separators=(', ', ': ')))
labels = {}
for instance in all["annotations"]:
    labels[instance.get("image_name")] = instance.get("bbox")

initial = INITIAL(root_path, dataset_name, ["BOX2D"], classes)
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train&test")
imgsName = os.listdir(os.path.join(root_path, "images"))
for img in imgsName:
    img_path = os.path.join(root_path, "images\\" + img)
    data = Data(img_path)
    label = labels[img]
    if len(label) != 0:
        data.label.box2d = []
        for ii in range(len(label)):
            ymin = label[ii].get("top")
            xmin = label[ii].get("left")
            w = label[ii].get("width")
            h = label[ii].get("height")
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmin + w, ymin + h,
                                                 category=splits[label[ii].get("class")],
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12, skip_uploaded_files=True)
dataset_client.commit("Initial commit")