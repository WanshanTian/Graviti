from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D
import os
from common.label_acquire import acquire_label_xml
from common.dataset_initial import initial

root_path = "G:\\shannon\\deeplearning_dataset\\playing-card"
dataset_name = "playing-card"
initial = initial(root_path, dataset_name, "box2d", "queen", "ten", "nine", "king", "jack", "ace")
gas, dataset = initial.generate_catalog()

for k in ["train", "test"]:
    segment = dataset.create_segment(k)
    files = os.listdir(os.path.join(root_path, k))
    imgs = []
    for file in files:
        if file.split(".")[-1] == "jpg" or file.split(".")[-1] == "JPG":
            imgs.append(file)
    for img_path in [os.path.join(os.path.join(root_path, k), i) for i in imgs]:
        label_path = img_path.split(".")[0] + ".xml"
        img_label = acquire_label_xml(label_path)
        data = Data(img_path)
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
