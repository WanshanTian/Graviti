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

keypoints = [{
    "number": 24,
    # "parentCategories": ["face_keypoints"],
    "names": [str(i) for i in list(range(24))],
    "skeleton": [],
    "visible": "BINARY"
}]

root_path = "G:\\download_dataset\\StanfordExtra"
label_path = os.path.join(root_path, "StanfordExtra_v1.json")

config.timeout = 100
config.max_retries = 10
dataset_name = "StanfordExtra"

classes = [x.split("-")[1] for x in os.listdir(os.path.join(root_path, "images"))]

initial = INITIAL(root_path, dataset_name, ["CLASSIFICATION", "KEYPOINTS2D", "BOX2D"], classes)
gas, dataset = initial.generate_catalog(keypoints)
with open(label_path, encoding='utf-8') as f:
    line = f.readline()
    all = json.loads(line)
img_labels = {}
for img in all:
    img_labels[img.get("img_path").split("/")[-1]] = [img.get("img_bbox"), img.get("joints"),
                                                      img.get("img_path").split("/")[0].split("-")[1]]

for split in os.listdir(os.path.join(root_path, "images")):
    segment = dataset.create_segment(split)
    imgsNames = os.listdir(os.path.join(root_path, "images" + "\\" + split))
    for imgName in imgsNames:
        img_path = os.path.join(os.path.join(root_path, "images" + "\\" + split), imgName)
        data = Data(img_path)

        if imgName in img_labels.keys():

            data.label.keypoints2d = []
            keypoints = LabeledKeypoints2D()
            for keypoint in range(len(img_labels[imgName][1])):
                x, y, v = img_labels[imgName][1][keypoint][0], img_labels[imgName][1][keypoint][1], int(
                    img_labels[imgName][1][keypoint][2])
                if v != 0:
                    keypoints.append(
                        Keypoint2D(x, y, v))
            data.label.keypoints2d.append(keypoints)
        # 分类
            data.label.classification = Classification(img_labels[imgName][2])
            # bbox
            data.label.box2d = []
            xmin = img_labels[imgName][0][0]
            ymin = img_labels[imgName][0][1]
            xmax = img_labels[imgName][0][2] + xmin
            ymax = img_labels[imgName][0][3] + ymin
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=img_labels[imgName][2]))
            segment.append(data)
        else:
            segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
