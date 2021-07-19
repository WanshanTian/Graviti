#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from tensorbay import GAS
from tensorbay.dataset import Dataset
from config import *

gas = GAS(access_key)  # create gas client

# 通过数据集名字获取需要更新数据集的dataset_client
dataset_client = gas.get_dataset("Winegrape")

# 创建draft commit
# dataset_client.create_draft("draft-1")

dataset = Dataset(dataset_name)

# 如果标签表发生改动，需要更新标签表
# dataset 是更新label后的 Dataset instance
# dataset_client.upload_catalog(dataset.catalog)

# 上传数据集中的所有label更新线上数据集
import os
with open(os.path.join(root_path, "train" + ".txt")) as f:
    ctx = f.readlines()
    imgs = [i.strip("\n") for i in ctx]
    labels = [i + ".txt" for i in imgs]
    imgs_name = [i + ".jpg" for i in imgs]
    image_files_path = [os.path.join(os.path.join(root_path, "data"), i) for i in imgs_name]
    label_files_path = [os.path.join(os.path.join(root_path, "data"), i) for i in labels]
for segment in dataset:
    segment_client = dataset_client.get_segment(segment.name)
    print(segment.name)
    for data in segment:
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
        segment_client.upload_label(data)

# commit
# dataset_client.commit("update label", tag="1.1")
