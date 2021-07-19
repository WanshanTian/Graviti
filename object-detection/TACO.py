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

root_path = "G:\\shannon\\deeplearning_dataset\\TACO\\TACO"
dataset_name = "TACO"

file_path= "G:\\shannon\\deeplearning_dataset\\TACO\TACO\\batch_1\\annotations.json"
with open(file_path, encoding='utf-8') as f:
    line = f.readline()
    all = json.loads(line)

# supercategories= []
subcategories = []
categories_origin = all.get("categories")
for i in categories_origin:
    if i.get('name') not in subcategories:
        subcategories.append(i.get('name'))

initial = initial(root_path, dataset_name, "box2d", subcategories)
gas, dataset = initial.generate_catalog()

for split in range(15):
    segment = dataset.create_segment("batch_"+str(split+1))
    # file_path = "G:\\shannon\\deeplearning_dataset\\TACO\TACO\\batch_1\\annotations.json"
    file_path = os.path.join(os.path.join(root_path,"batch_"+str(split+1)),"annotations.json")
    with open(file_path, encoding='utf-8') as f:
        line = f.readline()
        all = json.loads(line)

    img_id = {}
    for i in all["images"]:
        if i["file_name"] not in img_id.keys():
            img_id[i["file_name"]] = i["id"]

    category_id = {}
    for i in all["categories"]:
        if i["id"] not in category_id.keys():
            category_id[i["id"]] = i["name"]
            # category_id[i["id"]][]

    labels_dict = {}
    for i in all["annotations"]:
        if i["image_id"] not in labels_dict.keys():
            labels_dict[i["image_id"]] = []
        bbox = i["bbox"]
        cate_id = i["category_id"]
        label = bbox + [category_id[cate_id]]
        labels_dict[i["image_id"]].append(label)

    imgs_name = os.listdir(os.path.join(root_path, "batch_"+str(split+1)))
    imgs_name.remove("annotations.json")
    # print(imgs_name)
    for img_name in imgs_name:
        id = img_id[img_name]
        img_label = labels_dict[id]

        data = Data(os.path.join(os.path.join(root_path, "batch_"+str(split+1)), img_name))
        if len(img_label) != 0:
            data.label.box2d = []
            for i in range(len(img_label)):
                xmin = img_label[i][0]
                ymin = img_label[i][1]
                xmax = img_label[i][2]
                ymax = img_label[i][3]
                data.label.box2d.append(LabeledBox2D.from_xywh(xmin, ymin, xmax, ymax,
                                                     category=img_label[i][4],
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        segment.append(data)

dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")


