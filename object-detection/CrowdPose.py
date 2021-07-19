from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json

config.timeout = 40
config.max_retries = 4

root_path = "E:\\ShannonT\\dL-datasets\\CrowdPose\\CrowdPose_images"
dataset_name = "CrowdPose"

imgs = os.listdir(os.path.join(root_path, "images"))

initial = initial(root_path, dataset_name, ["BOX2D", "KEYPOINTS2D"], ["person"])
gas, dataset = initial.generate_catalog()

splits = ["train", "test", "val"]

for split in splits:
    label_file_path = os.path.join("E:\\ShannonT\\dL-datasets\\CrowdPose\\CrowdPose_annotations\\json",
                                   "crowdpose_" + split + ".json")
    segment = dataset.create_segment(split)
    with open(label_file_path, encoding='utf-8') as f:
        line = f.readline()
        all = json.loads(line)
    # labels["id"]=[[bbox],[keypoints]]
    labels = {}
    for ele in all.get("annotations"):
        if ele["image_id"] not in labels.keys():
            labels[ele["image_id"]] = []
        labels[ele["image_id"]].append([ele["bbox"], ele["keypoints"]])
    for i in all.get("images"):
        img_name = i.get("file_name")
        # crowdIndex = i.get("crowdIndex")
        img_id = i.get("id")
        img_path = os.path.join(os.path.join(root_path, "images"), img_name)
        data = Data(img_path)
        img_label = labels[img_id]
        if len(img_label) != 0:
            data.label.box2d = []
            data.label.keypoints2d = []
            for ii in range(len(img_label)):
                xmin = img_label[ii][0][0]
                ymin = img_label[ii][0][1]
                xmax = img_label[ii][0][2]
                ymax = img_label[ii][0][3]
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmin + xmax, ymin + ymax,
                                                     category="person",
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
                # 添加keypoints2D标签
                keypoints = LabeledKeypoints2D(category="person")
                for keypoint in range(14):
                    keypoints.append(
                        Keypoint2D(img_label[ii][1][3 * keypoint], img_label[ii][1][3 * keypoint + 1], img_label[ii][1][3 * keypoint + 2]))
                data.label.keypoints2d.append(keypoints)
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
