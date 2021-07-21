from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json

keypoints = [{
    "number": 25,
    "parentCategories": ["pose_keypoints_2d"],
    "names": [str(i) for i in list(range(25))],
    "skeleton": [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [1, 5], [8, 12],
                 [2, 9], [5, 12]],
    "visible": "BINARY"
}, {
    "number": 70,
    "parentCategories": ["face_keypoints_2d"],
    "names": [str(i) for i in list(range(70))],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                 [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24],
                 [24, 25], [25, 26], [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35], [36, 37],
                 [37, 38], [38, 39], [39, 40], [40, 41], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [48, 49],
                 [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59],
                 [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [36, 41], [42, 47], [48, 59],
                 [60, 67]],
    "visible": "BINARY"
}, {
    "number": 21,
    "parentCategories": ["hand_left_keypoints_2d"],
    "names": [str(i) for i in list(range(21))],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14],
                 [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]],
    "visible": "BINARY"
}, {
    "number": 21,
    "parentCategories": ["hand_right_keypoints_2d"],
    "names": [str(i) for i in list(range(21))],
    "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14],
                 [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]],
    "visible": "BINARY"
}]

config.timeout = 40
config.max_retries = 4

root_path = "G:\download_dataset\EHF\EHF"
dataset_name = "EHF"

imgs_catalog = os.path.join(root_path, "EHF")
labels_catalog = os.path.join(os.path.join(root_path, "bioid_pts"), "points_20")

imgs_fileName = [x for x in os.listdir(imgs_catalog) if x.endswith(".jpg")]

initial = INITIAL(root_path, dataset_name, ["KEYPOINTS2D"],
                  ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"])
gas, dataset = initial.generate_catalog(keypoints)

segment = dataset.create_segment("train&test")
for img in imgs_fileName:
    img_path = os.path.join(root_path, "EHF\\" + img)
    label_path = os.path.join(root_path, "EHF\\" + img.split("_")[0] + "_2Djnt.json")
    with open(label_path, encoding='utf-8') as f:
        line = f.readline()
        labels = json.loads(line)
    labels = labels["people"][0]
    data = Data(img_path)
    data.label.keypoints2d = []
    for kind in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
        keypoints = LabeledKeypoints2D(category=kind)
        for keypoint in range(int(len((labels[kind])) / 3)):
            x, y = labels[kind][3 * keypoint], labels[kind][3 * keypoint + 1]
            keypoints.append(
                Keypoint2D(x, y))
        data.label.keypoints2d.append(keypoints)
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
#
