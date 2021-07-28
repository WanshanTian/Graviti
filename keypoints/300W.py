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

root_path = "G:\download_dataset\ThreeHundredW\\300w"
config.timeout = 400
config.max_retries = 10

dataset_name = "ThreeHundredW"
keypoints = [{
    "number": 68,
    # "parentCategories": ["face_keypoints"],
    "names": [str(i) for i in list(range(68))],
    "skeleton": [],
    "visible": "BINARY"
}]

initial = INITIAL(root_path, dataset_name, ["KEYPOINTS2D"], ["face_keypoints"])
gas, dataset = initial.generate_catalog(keypoints)

for split in ["01_Indoor", "02_Outdoor"]:
    segment = dataset.create_segment(split)
    imgName = [x for x in os.listdir(os.path.join(root_path, split)) if x.endswith(".png")]
    for img in imgName:
        img_path = os.path.join(root_path, split + "\\" + img)
        label_path = os.path.join(root_path, split + "\\" + img.split(".")[0] + ".pts")
        l = open(label_path, 'rb').read()
        label = bytes.decode(l)
        po = label.split("{")[1].split("}")[0]
        img_label = po[1:].split("\n")[:-1]
        data = Data(img_path)

        data.label.keypoints2d = []
        keypoints = LabeledKeypoints2D(category="face_keypoints")
        for label in img_label:
            x = float(label.split(" ")[0])
            y = float(label.split(" ")[1])
            keypoints.append(Keypoint2D(x, y))
        data.label.keypoints2d.append(keypoints)
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=8)
dataset_client.commit("Initial commit")

