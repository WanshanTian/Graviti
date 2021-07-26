from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
import csv

root_path = "G:\\download_dataset\\OTB100\\OTB100"
config.timeout = 100
config.max_retries = 8

dataset_name = "OTB100"

splits = os.listdir(root_path)

initial = INITIAL(root_path, dataset_name, ["BOX2D"], splits)
gas, dataset = initial.generate_catalog()

if "catalog.json" in splits:
    splits.remove("catalog.json")
for split in splits:
    segment = dataset.create_segment(split)
    imgsName = [x for x in os.listdir(os.path.join(root_path, split + "\\" + "img")) if x.endswith(".jpg")]
    if split == "Skating2":
        labels_path = os.path.join(root_path, split + "\\" + "groundtruth_rect.1.txt")
    else:
        labels_path = os.path.join(root_path, split + "\\" + "groundtruth_rect.txt")
    labels_origin = []

    with open(labels_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            labels_origin.append(l.strip("\n"))
    for img in range(len(imgsName)):
        img_path = os.path.join(os.path.join(root_path, split + "\\" + "img"), imgsName[img])
        data = Data(img_path)

        print(split)

        if split == "David":

            if img >= 300:
                img_label = [float(x) for x in labels_origin[img - 300].split(",")]
                data.label.box2d = []
                xmin = img_label[0]
                ymin = img_label[1]
                xmax = img_label[2] + xmin
                ymax = img_label[3] + ymin
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=split,
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        elif split == "Football1":
            if img < 74:
                img_label = [float(x) for x in labels_origin[img].split(",")]
                data.label.box2d = []
                xmin = img_label[0]
                ymin = img_label[1]
                xmax = img_label[2] + xmin
                ymax = img_label[3] + ymin
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=split,
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        elif split == "Jogging" or split == "Rubik" or split=="Toy" or split == "Vase" or split=="Walking2" or split ==  "Walking" or split== "Twinnings" or split=="Sylvester" or split == "Singer1" or split == "Skating2" or split == "Subway" or split == "Surfer":

            img_label = [float(x) for x in labels_origin[img].split(" ")]
            data.label.box2d = []
            xmin = img_label[0]
            ymin = img_label[1]
            xmax = img_label[2] + xmin
            ymax = img_label[3] + ymin
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=split,
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
        elif split == "Jogging" or split == "Rubik" or split == "Singer1":

            img_label = [float(x) for x in labels_origin[img].split(" ")]
            data.label.box2d = []
            xmin = img_label[0]
            ymin = img_label[1]
            xmax = img_label[2] + xmin
            ymax = img_label[3] + ymin
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=split,
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
        elif split == "Diving":
            i = 1
        elif split == "Freeman3":
            if img < 460:
                img_label = [float(x) for x in labels_origin[img].split(",")]
                data.label.box2d = []
                xmin = img_label[0]
                ymin = img_label[1]
                xmax = img_label[2] + xmin
                ymax = img_label[3] + ymin
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=split,
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        elif split == "Freeman4":
            if img < 283:
                img_label = [float(x) for x in labels_origin[img].split(",")]
                data.label.box2d = []
                xmin = img_label[0]
                ymin = img_label[1]
                xmax = img_label[2] + xmin
                ymax = img_label[3] + ymin
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=split,
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))

        else:
            try:
                img_label = [float(x) for x in labels_origin[img].split(",")]
            except:
                img_label = [float(x) for x in labels_origin[img].split(" ")]
            if len(img_label) != 0:
                data.label.box2d = []
                xmin = img_label[0]
                ymin = img_label[1]
                xmax = img_label[2] + xmin
                ymax = img_label[3] + ymin
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=split,
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=6)
dataset_client.commit("Initial commit")
