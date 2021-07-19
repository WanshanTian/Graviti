from common.dataset_initial import initial
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

gas = GAS(access_key)
dataset_name = "BioIDFace-test"
dataset_client = gas.get_dataset(dataset_name)
draft_number = dataset_client.create_draft("draft-" + str(datetime.datetime.now()))

dataset_client.delete_segment("train&test")

dataset_client.commit("commit-"+str(datetime.datetime.now()), "commit description")

