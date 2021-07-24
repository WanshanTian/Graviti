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

root_path = "G:\\download_dataset\\FacialKeypointDetection"
config.timeout = 40
config.max_retries = 4

dataset_name = "FacialKeypointDetection"

keypoints = [{
    "number": 15,
    "parentCategories": ["face_keypoints"],
    "names": ['left_eye_center', 'right_eye_center',
              'left_eye_inner_corner', 'left_eye_outer_corner',
              'right_eye_inner_corner',
              'right_eye_outer_corner', 'left_eyebrow_inner_end',
              'left_eyebrow_outer_end',
              'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
              'nose_tip', 'mouth_left_corner',
              'mouth_right_corner', 'mouth_center_top_lip',
              'mouth_center_bottom_lip', ],
    "skeleton": [],
    "visible": "BINARY"
}]

with open(os.path.join(root_path, "training\\training.csv"), "r") as file:
    csv_file = csv.reader(file)
    csv_file.__next__()
    keypoints_xy = next(csv_file)
print(keypoints_xy[:30])

initial = INITIAL(root_path, dataset_name, ["KEYPOINTS2D"], ["face_keypoints"])
gas, dataset = initial.generate_catalog(keypoints)

for split in ["training", "test"]:
    segment = dataset.create_segment(split)
    imgs_catalog = os.path.join(root_path, split)
    imgs_fileName = [x for x in os.listdir(imgs_catalog) if x.endswith(".png")]
    labels = []
    with open(os.path.join(root_path, split + "\\" + split + ".csv"), "r") as file:
        csv_file = csv.reader(file)
        csv_file.__next__()
        for row in csv_file:
            if split == "training":
                labels.append(row[0:30])
    for img in imgs_fileName:
        img_path = os.path.join(root_path, split + "\\" + img)
        data = Data(img_path)
        if len(labels) != 0:
            label = labels[int(img.split(".")[0])]
            for i in range(len(label)):
                if label[i] == "":
                    label[i] = 0
            label_new = [float(x) for x in label]

            data.label.keypoints2d = []

            keypoints = LabeledKeypoints2D(category="face_keypoints")
            for keypoint in range(int(len(label_new) / 2)):
                x, y = label_new[keypoint * 2], label_new[keypoint * 2 + 1]
                keypoints.append(
                    Keypoint2D(x, y))
            data.label.keypoints2d.append(keypoints)
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
