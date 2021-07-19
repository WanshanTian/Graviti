from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D
import os
import cv2

gas = GAS(access_key)
# gas.create_dataset(dataset_name)
dataset = Dataset(dataset_name)
dataset.load_catalog(os.path.join(root_path, "catalog.json"))
for k, v in LABEL_FILENAME_DICT.items():
    segment = dataset.create_segment(k)
    with open(os.path.join(root_path, k + ".txt")) as f:
        ctx = f.readlines()
        imgs = [i.strip("\n") for i in ctx]
        labels = [i + ".txt" for i in imgs]
        imgs_name = [i + ".jpg" for i in imgs]
    image_files_path = [os.path.join(os.path.join(root_path, "data"), i) for i in imgs_name]
    label_files_path = [os.path.join(os.path.join(root_path, "data"), i) for i in labels]

    # 读取标签
    for i in range(len(image_files_path)):
        data = Data(image_files_path[i])
        img = cv2.imread(image_files_path[i])
        width = img.shape[1]
        height = img.shape[0]
        # label = labels[img_path.split("\\")[-1]]
        label_path = label_files_path[i]
        with open(label_path, "r") as file:
            txt_file = file.readlines()
            labels = [i.strip("\n") for i in txt_file]
        data.label.box2d = []
        for label in labels:
            cx = int(float(label.split(" ")[1]) * width)
            cy = int(float(label.split(" ")[2]) * height)
            w = int(float(label.split(" ")[3]) * width)
            h = int(float(label.split(" ")[4]) * height)
            data.label.box2d.append(LabeledBox2D.from_xywh(int(cx - w / 2), int(cy - h / 2), w, h,
                                                           category=imgs[i].split("_")[0],
                                                           # attributes={"occluded": box["occluded"]}))
                                                           ))
        segment.append(data)
dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")
