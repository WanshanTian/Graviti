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

root_path = "G:\download_dataset\VOC2012PersonLayout\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
config.timeout = 400
config.max_retries = 10

dataset_name = "VOC2012PersonLayout"

classes = ['person', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog', 'chair', 'bird', 'bicycle', 'bottle', 'sheep',
           'diningtable', 'horse', 'motorbike', 'sofa', 'cow', 'car', 'cat', 'bus', 'pottedplant']
initial = INITIAL(root_path, dataset_name, ["BOX2D"], classes)
gas, dataset = initial.generate_catalog()
imgsName = os.listdir(os.path.join(root_path, "JPEGImages"))

segment = dataset.create_segment("train&test")
for img in imgsName:
    img_label = acquire_label_xml(os.path.join(root_path, "Annotations\\" + img.split(".")[0] + ".xml"))
    img_path = os.path.join(root_path , "JPEGImages\\"+img)
    data = Data(img_path)
    if len(img_label) != 0:
        data.label.box2d = []
        data.label.keypoints2d = []
        for ii in range(len(img_label)):
            xmin = img_label[ii][0]
            ymin = img_label[ii][1]
            xmax = img_label[ii][2]
            ymax = img_label[ii][3]
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=img_label[ii][4],
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")


