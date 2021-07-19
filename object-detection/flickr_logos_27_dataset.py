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
import json

root_path = "G:\\shannon\\deeplearning_dataset\\flickr_logos_27_dataset"
dataset_name = "flickr_logos_27_dataset"
# labels = os.listdir(os.path.join(root_path, "Annotations"))
imgs = os.listdir(os.path.join(root_path, "flickr_logos_27_dataset_images"))

label_path = os.path.join(root_path, "flickr_logos_27_dataset_training_set_annotation.txt")
test_path = os.path.join(root_path, "flickr_logos_27_dataset_query_set_annotation.txt")

labels_origin = []
with open(label_path, "r") as f:
    for line in f.readlines():
        labels_origin.append(line.strip("\n"))

classes = []
imgs_label = {}
for i in labels_origin:
    if i.split(" ")[1] not in classes:
        classes.append(i.split(" ")[1])
    if i.split(" ")[0] not in imgs_label.keys():
        imgs_label[i.split(" ")[0]] = []
    if [int(i.split(" ")[3]), int(i.split(" ")[4]), int(i.split(" ")[5]), int(i.split(" ")[6]), i.split(" ")[1]] not in imgs_label[i.split(" ")[0]]:
        imgs_label[i.split(" ")[0]].append(
            [int(i.split(" ")[3]), int(i.split(" ")[4]), int(i.split(" ")[5]), int(i.split(" ")[6]), i.split(" ")[1]])

initial = initial(root_path, dataset_name, "box2d", classes)
gas, dataset = initial.generate_catalog()

for i in ["train", "test"]:
    if i == "train":
        segment = dataset.create_segment(i)
        for j in list(imgs_label.keys()):
            path = os.path.join(os.path.join(root_path, "flickr_logos_27_dataset_images"), j)
            data = Data(path)
            if j not in imgs_label.keys():
                img_label = []
            else:
                img_label = imgs_label[j]
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
    elif i == "test":
        segment = dataset.create_segment(i)
        test_imgs = []
        for ii in imgs:
            if ii not in list(imgs_label.keys()):
                test_imgs.append(ii)
        for j in test_imgs:
            path = os.path.join(os.path.join(root_path, "flickr_logos_27_dataset_images"), j)
            data = Data(path)
            segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
