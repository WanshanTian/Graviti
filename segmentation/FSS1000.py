from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D, SemanticMask, InstanceMask
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

dataset_name = "FSS1000"
root_path = "D:\\download_dataset\\FSS1000\\fewshot_data"

classes = sorted(os.listdir(os.path.join(root_path, "fewshot_data")))

initial = INITIAL(root_path, dataset_name, ["CLASSIFICATION", "SEMANTIC_MASK"], classes)
gas, dataset = initial.generate_catalog()

for split in classes:
    segment = dataset.create_segment(split)
    imgsName = [x for x in os.listdir(os.path.join(root_path, "fewshot_data\\" + split)) if x.endswith(".jpg")]
    for img in imgsName:
        img_path = os.path.join(root_path, "fewshot_data\\" + split + "\\" + img)
        mask_path = os.path.join(root_path, "fewshot_data\\" + split + "\\" + img.split(".")[0] + "mask.png")
        data = Data(img_path)
        data.label.classification = Classification(split)
        data.label.semantic_mask = SemanticMask(mask_path)
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
