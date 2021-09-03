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

config.timeout = 40
config.max_retries = 4

dataset_name = "DeepPCB"
root_path = "G:\\download_dataset\\DeepPCB\\DeepPCB\\DeepPCB-master\\PCBData"

initial = INITIAL(root_path, dataset_name, ["BOX2D"],
                  ["open", "short", "mousebite", "spur", "pin hole", "spurious copper"])
gas, dataset = initial.generate_catalog()

splits = ["trainval", "test"]

classes = {}
for i, j in zip(range(6), ["open", "short", "mousebite", "spur", "copper", "pin-hole"]):
    classes[i+1] = j

for split in splits:
    segment = dataset.create_segment(split)
    file_content = []
    train_file_path = os.path.join(root_path, split + ".txt")
    with open(train_file_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            file_content.append(l.strip("\n"))
    file_path = [i.split(" ")[0] for i in file_content]
    label_path = [i.split(" ")[1] for i in file_content]
    for img, label in zip(file_path, label_path):
        img_path = os.path.join(root_path, img)
        img_path = img_path.replace("/", "\\")
        img_path = img_path.split(".")[0]+"_test.jpg"
        label_path_full = os.path.join(root_path, label)
        label_path_full = label_path_full.replace("/", "\\")
        file_name = img.split("/")[-1]
        # 获取label
        img_label = []
        with open(label_path_full, "r") as f:
            for line in f.readlines():
                l = " ".join(line.split("\t"))
                img_label.append(l.strip("\n"))
        data = Data(img_path)
        if len(img_label) != 0:
            data.label.box2d = []
            for ii in range(len(img_label)):
                tmp = [float(p) for p in img_label[ii].split(" ")]
                xmin = float(tmp[0])
                ymin = float(tmp[1])
                xmax = float(tmp[2])
                ymax = float(tmp[3])
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=classes[tmp[4]],
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
