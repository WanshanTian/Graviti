from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D
import os
from common.label_acquire import acquire_label_xml
from common.dataset_initial import initial
import os
import csv

root_path = "G:\\shannon\\deeplearning_dataset\\nexet\\nexet"
dataset_name = "NEXET"
initial = initial(root_path, dataset_name, "box2d", 'van', 'car', 'truck', 'bus', 'pickup_truck')
gas, dataset = initial.generate_catalog()

imgs_lable_dict = {}
with open(os.path.join(root_path, "train_boxes.csv"), "r") as file:
    csv_file = csv.reader(file)
    csv_file.__next__()
    for row in csv_file:
        if row[0] not in imgs_lable_dict.keys():
            imgs_lable_dict[row[0]] = []
        imgs_lable_dict[row[0]].append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), row[5]])

segment = dataset.create_segment("train_50k")
files = os.listdir(os.path.join(root_path, "nexet_2017_1"))
# imgs_path = [os.path.join(os.path.join(root_path, "nexet_2017_1"),i) for i in files]

for img in files:
    if img in imgs_lable_dict.keys():
        img_label = imgs_lable_dict[img]
    else:
        img_label = []
    data = Data(os.path.join(os.path.join(root_path, "nexet_2017_1"), img))
    if len(img_label) != 0:
        data.label.box2d = []
        for i in range(len(img_label)):
            xmin = img_label[i][0]
            ymin = img_label[i][1]
            xmax = img_label[i][2]
            ymax = img_label[i][3]
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=img_label[i][4],
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
    segment.append(data)

dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")
