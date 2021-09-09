from common.dataset_initial import INITIAL, update_catalog
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
import datetime
import time
from tensorbay import GAS
from category import catalog
from config import access_key
from tensorbay.dataset import Dataset

# # config.timeout = 400
# # config.max_retries = 10
# gas = GAS(access_key)
# # dataset_name = "MSRA10K-test"
#
# root_path = "G:\download_dataset\DatasetForInfraredImageDimSmallAircraftTargetDetectionAndTrackingUnderGroundAirBackground"
# dataset_name = "DatasetForInfraredImageDimSmallAircraftTargetDetectionAndTrackingUnderGroundAirBackground"
# dataset_client = gas.get_dataset(dataset_name)
# for i in list(range(1, 23)):
#
#     draft_number = dataset_client.create_draft("draft-" + str(datetime.datetime.now()))
#     dataset_client.delete_segment("data"+str(i))
#     dataset_client.commit("commit-" + str(datetime.datetime.now()), "commit description")

# root_path = "G:\download_dataset\DatasetForInfraredImageDimSmallAircraftTargetDetectionAndTrackingUnderGroundAirBackground"
# dataset_name = "TrackingUnderGroundAirBackground"
# imgName = os.listdir(os.path.join(root_path, "Images"))
# dataset = Dataset(dataset_name)
# segment_client = dataset_client.get_segment("train&test")
# segment_client.delete_data(imgName)


# root_path = "G:\download_dataset\VOC2012PersonLayout\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"
# dataset_name = "VOC2012PersonLayout"
# classes = ['person', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog', 'chair', 'bird', 'bicycle', 'bottle', 'sheep',
#            'diningtable', 'horse', 'motorbike', 'sofa', 'cow', 'car', 'cat', 'bus', 'pottedplant']
# update_catalog(root_path, dataset_name , ["BOX2D"], classes)


gas = GAS(access_key)
dataset_client = gas.get_dataset("FurgFire")

dataset_client.create_draft("draft-" + str(datetime.datetime.now()))

for segment_name in dataset_client.list_segment_names():
    segment_client = dataset_client.get_segment(segment_name)
    data_remote_paths = list(segment_client.list_data_paths())
    new_paths = []
    for path in data_remote_paths:
        new_path, num = path.rsplit("_", 1)
        print(new_path, num)
        new_paths.append(f"{new_path}_{num.zfill(9)}")

    segment_client.move_data(data_remote_paths, new_paths)

dataset_client.commit("change remote path")
