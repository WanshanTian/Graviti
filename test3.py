from common.dataset_initial import INITIAL
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
config.timeout = 400
config.max_retries = 10
gas = GAS(access_key)
dataset_name = "SCUT_FBP5500"
root_path = "G:\download_dataset\SCUT_FBP5500\SCUT-FBP5500_v2.1\SCUT-FBP5500_v2"
imgName = os.listdir(os.path.join(root_path, "Images"))
dataset = Dataset(dataset_name)
dataset_client = gas.get_dataset(dataset_name)
draft_number = dataset_client.create_draft("draft-" + str(datetime.datetime.now()))

# dataset_client.delete_segment("train&test")
segment_client = dataset_client.get_segment("train&test")
segment_client.delete_data(imgName)


dataset_client.commit("commit-"+str(datetime.datetime.now()), "commit description")

