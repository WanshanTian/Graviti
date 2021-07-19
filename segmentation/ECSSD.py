import os

from tensorbay.client import config
from tensorbay.dataset import Data

from common.dataset_initial import INITIAL

config.timeout = 40
config.max_retries = 4

dataset_name = "ECSSD"
root_path = "G:\\download_dataset\\ECSSD"

imgs_fileName = [x for x in os.listdir(os.path.join(root_path, "images")) if x.endswith(".jpg")]
masks_fileName = [x for x in os.listdir(os.path.join(root_path, "ground_truth_mask")) if x.endswith(".png")]

initial = INITIAL(root_path, dataset_name, [], [])
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train&test")
for img_fileName in imgs_fileName:
    img_path = os.path.join(root_path, "images\\" + img_fileName)
    data = Data(img_path)
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
