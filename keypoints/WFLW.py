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

root_path = "G:\download_dataset\WFLW"
config.timeout = 400
config.max_retries = 10
keypoints = [{
    "number": 98,
    # "parentCategories": ["face_keypoints"],
    "names": [str(i) for i in list(range(98))],
    "skeleton": [],
    "visible": "BINARY"
}]
dataset_name = "WFLW"

classes = [x.split("--")[1] for x in os.listdir(os.path.join(root_path, "WFLW_images"))]
#
initial = INITIAL(root_path, dataset_name, ["BOX2D", "KEYPOINTS2D"], classes)
gas, dataset = initial.generate_catalog(keypoints)

for split in ["train", "test"]:
    segment = dataset.create_segment(split)
    label_path = os.path.join(root_path,
                              "WFLW_annotations\\list_98pt_rect_attr_train_test\\" + "list_98pt_rect_attr_" + split + ".txt")
    labels_origin = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            labels_origin.append(l.strip("\n"))
    # print(labels_origin[0])
    for img in labels_origin:
        img_path = os.path.join(root_path, "WFLW_images\\" + img.split(" ")[-1])
        label = [float(x) for x in img.split(" ")[:-11]]
        bbox = [float(x) for x in img.split(" ")[-11:-7]]
        att = [float(x) for x in img.split(" ")[-7:-1]]
        data = Data(img_path)

        data.label.keypoints2d = []
        keypoints = LabeledKeypoints2D()
        for i in range(int(len(label) / 2)):
            x = label[2 * i]
            y = label[2 * i + 1]
            keypoints.append(Keypoint2D(x, y))
        data.label.keypoints2d.append(keypoints)
        # bbox
        data.label.box2d = []
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                             category=img.split(" ")[-1].split("/")[0].split("--")[1],
                                             # attributes={"pose": att[0], "expression": att[1],
                                             #             "illumination": att[2], "make-up": att[3],
                                             #             "occlusion": att[4], "blur": att[5]}
                                                         ))

        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
