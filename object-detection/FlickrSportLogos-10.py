from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification
from tensorbay.dataset import Data
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file

config.timeout = 40
config.max_retries = 4

root_path = "E:\\ShannonT\\dL-datasets\\FlickrSportLogos-10-master\\dataset"
dataset_name = "FlickrSportLogos-10"

files = os.listdir(os.path.join(root_path, "JPEGImages"))
classes = []
for img in files:
    if img.split("_")[0] not in classes:
        classes.append(img.split("_")[0])

initial = initial(root_path, dataset_name, ["CLASSIFICATION", "BOX2D"], classes)
gas, dataset = initial.generate_catalog()

for i in ["train.txt", "test.txt"]:
    segment = dataset.create_segment(i.split(".")[0])
    imgs = []
    imgs_path = os.path.join(root_path, i)
    with open(imgs_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            imgs.append(l.strip("\n"))
    for img in imgs:
        path = os.path.join(os.path.join(root_path, "JPEGImages"), img + ".jpg")
        label_path = os.path.join(os.path.join(root_path, "Annotations"), img + ".xml")
        img_label = acquire_label_xml(label_path)
        data = Data(path)
        cate=img.split("_")[0]
        data.label.classification = Classification(cate)
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

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
