from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data, Dataset
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
from tensorbay import GAS
from config import access_key

config.timeout = 40
config.max_retries = 4

dataset_name = "DatasetForInfraredImageDimSmallAircraftTargetDetectionAndTrackingUnderGroundAirBackground"
# dataset_name = "TrackingUnderGroundAirBackground"
root_path = "G:\download_dataset\DatasetForInfraredImageDimSmallAircraftTargetDetectionAndTrackingUnderGroundAirBackground"

classes = ["data" + str(x + 1) for x in list(range(22))]
keypoints = [{
    "number": 1,
    # "parentCategories": ["object1"],
    "names": [str(i) for i in list(range(1))],
    "skeleton": [],
    "visible": "BINARY"
}, ]
initial = INITIAL(root_path, dataset_name, ["KEYPOINTS2D"], ["object"])
gas, dataset = initial.generate_catalog(keypoints)

for split in classes:
    segment = dataset.create_segment("data" + "%02d" % int(split.split("data")[1]))
    imgNames = [x for x in os.listdir(os.path.join(root_path, split)) if x.endswith(".jpg")]
    # acquire labels
    label_path = os.path.join(root_path, "data_label\\" + split + ".txt")
    labels_origin = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            l = " ".join(line.split("\t"))
            labels_origin.append(l.strip("\n"))
    keypointsall = []
    for i in labels_origin[1:]:
        keypoint = []
        for j in range(len(i.split())):
            if i.split()[j].startswith("object"):
                keypoint.append(int(i.split()[j + 1]))
                keypoint.append(int(i.split()[j + 2]))
        keypointsall.append(keypoint)
    for img in imgNames:
        imgPath = os.path.join(root_path, split + "\\" + img)
        data = Data(imgPath, target_remote_path="%04d" % int(img.split(".")[0]) + ".jpg")
        # add label
        data.label.keypoints2d = []

        label = keypointsall[int(img.split(".")[0])]
        for point in range(int(len(label) / 2)):
            # keypoints = LabeledKeypoints2D(category="object"+str(point+1))
            keypoints = LabeledKeypoints2D(category="object")
            x = label[point * 2]
            keypoints.append(Keypoint2D(x, y))
            data.label.keypoints2d.append(keypoints)
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
