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

dataset_name = "MSRA10K"
root_path = "G:\\download_dataset\\MSRA10K\\MSRA10K_Imgs_GT\\MSRA10K_Imgs_GT"

imgs_fileName = [x for x in os.listdir(os.path.join(root_path, "Imgs")) if x.endswith(".jpg")]
masks_fileName = [x for x in os.listdir(os.path.join(root_path, "Imgs")) if x.endswith(".png")]



initial = INITIAL(root_path, dataset_name, [], [])
gas, dataset = initial.generate_catalog()

if dataset_name not in list(gas.list_dataset_names()):
    gas.create_dataset(dataset_name)
dataset = Dataset(dataset_name)

segment = dataset.create_segment("train&test")
for img_fileName in imgs_fileName:
    img_path = os.path.join(root_path, "Imgs\\"+img_fileName)
    data = Data(img_path)
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")


