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

dataset_name = "PennFudanDatabaseForPedestrianDetectionAndSegmentation"
root_path = "G:\download_dataset\PennFudanDatabaseForPedestrianDetectionAndSegmentation\PennFudanPed"

imgsName = os.listdir(os.path.join(root_path, "PNGImages"))

initial = INITIAL(root_path, dataset_name, ["BOX2D"], ["Person"])
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train&test")
for img in imgsName:
    img_path = os.path.join(root_path, "PNGImages\\"+img)
    label_path = os.path.join(root_path, "Annotation\\" + img.split(".")[0] + ".txt")
    labels_origin = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            labels_origin.append(l.strip("\n"))
    bboxs = []
    for i in labels_origin:
        if i.startswith("Bounding"):
            bbox = []
            x1y1 = i.split(":")[-1].split("-")[0]
            x1 = float(x1y1.split("(")[1].split(",")[0])
            y1 = float(x1y1.split(")")[0].split(" ")[-1])
            x2y2 = i.split(":")[-1].split("-")[1]
            x2 = float(x2y2.split("(")[1].split(",")[0])
            y2 = float(x2y2.split(")")[0].split(" ")[-1])
            bbox.append(x1)
            bbox.append(y1)
            bbox.append(x2)
            bbox.append(y2)
            bboxs.append(bbox)
    data = Data(img_path)
    img_label = bboxs
    if len(img_label) != 0:
        data.label.box2d = []
        for ii in range(len(img_label)):
            xmin = img_label[ii][0]
            ymin = img_label[ii][1]
            xmax = img_label[ii][2]
            ymax = img_label[ii][3]
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category="Person",
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")

