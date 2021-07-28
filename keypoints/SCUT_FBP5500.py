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

root_path = "G:\download_dataset\SCUT_FBP5500\SCUT-FBP5500_v2.1\SCUT-FBP5500_v2"
config.timeout = 40
config.max_retries = 4

dataset_name = "SCUT_FBP5500"
keypoints = [{
    "number": 86,
    # "parentCategories": ["face_keypoints"],
    "names": [str(i) for i in list(range(86))],
    "skeleton": [],
    "visible": "BINARY"
}]
imgName = os.listdir(os.path.join(root_path, "Images"))

initial = INITIAL(root_path, dataset_name, ["KEYPOINTS2D"], ["face_keypoints"])
gas, dataset = initial.generate_catalog(keypoints)

segment = dataset.create_segment("train&test")
for img in imgName:
    img_path = os.path.join(root_path, "Images\\" + img)
    data = Data(img_path)
    try:
        img_label_path = os.path.join(root_path, "landmark_txt\\" + img.split(".")[0] + ".txt")
        labels_origin = []
        with open(img_label_path, "r") as f:
            for line in f.readlines():
                l = " ".join(line.split("\t"))
                labels_origin.append(l.strip("\n"))

        data.label.keypoints2d = []
        keypoints = LabeledKeypoints2D(category="face_keypoints")
        for label in labels_origin:
            x = float(label.split(" ")[0])
            y = float(label.split(" ")[1])
            keypoints.append(Keypoint2D(x, y))
        data.label.keypoints2d.append(keypoints)
        segment.append(data)
    except:
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")

