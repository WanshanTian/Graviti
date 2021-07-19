from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json

keypointsFace = {
    "number": 20,
    "names": ['right eye pupil',
              "left eye pupil",
              "right_mouth_corner",
              "left mouth corner",
              "outer end of right eye brow",
              "inner end of right eye brow",
              "inner end of left eye brow",
              "outer end of left eye brow",
              "right temple",
              "outer corner of right eye",
              "inner corner of right eye",
              "inner corner of left eye",
              "outer corner of left eye",
              "left temple",
              "tip of nose",
              "right nostril",
              "left nostril",
              "centre point on outer edge of upper lip",
              "centre point on outer edge of lower lip",
              "tip of chin"],
    "skeleton": [[4, 9], [9, 0], [0, 10], [10, 5], [6, 11], [11, 1], [1, 12], [12, 7], [15, 14], [14, 16],
                 [2, 17], [17, 3], [3, 18], [18, 2]],
    "visible": "BINARY"
}

config.timeout = 40
config.max_retries = 4

root_path = "G:\\download_dataset\\BioIDFace"
dataset_name = "BioIDFace-test"

imgs_catalog = os.path.join(root_path, "BioID-FaceDatabase-V1.2")
labels_catalog = os.path.join(os.path.join(root_path, "bioid_pts"), "points_20")

initial = initial(root_path, dataset_name, ["KEYPOINTS2D"], ["faceKeypoint"])
gas, dataset = initial.generate_catalog(keypointsFace)

segment = dataset.create_segment("train&test")

imgs_fileName = [x for x in os.listdir(imgs_catalog) if x.endswith(".pgm")]

for img in imgs_fileName:
    label_fileName = img.split(".")[0] + ".pts"
    label_path = os.path.join(labels_catalog, label_fileName)
    #获取points
    img_labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            img_labels.append(l.strip("\n"))
    img_label = img_labels[3:-1]
    #上传图像
    img_path = os.path.join(imgs_catalog, img)
    data = Data(img_path)
    data.label.keypoints2d = []
    keypoints = LabeledKeypoints2D(category="faceKeypoint")
    for keypoint in range(20):
        x, y = float(img_label[keypoint].split(" ")[0]), float(img_label[keypoint].split(" ")[1])
        keypoints.append(
            Keypoint2D(x, y))
    data.label.keypoints2d.append(keypoints)
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")

