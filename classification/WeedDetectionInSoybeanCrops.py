import os

from tensorbay.client import config
from tensorbay.dataset import Data
from tensorbay.label import Classification

from common.dataset_initial import INITIAL

root_path = "G:\download_dataset\WeedDetectionInSoybeanCrops\weed-detection-in-soybean-crops\dataset"
config.timeout = 400
config.max_retries = 10

dataset_name = "WeedDetectionInSoybeanCrops"

classes = os.listdir(root_path)
if "catalog.json" in classes:
    classes.remove("catalog.json")

initial = INITIAL(root_path, dataset_name, ["CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

for split in classes:
    segment = dataset.create_segment(split)
    imgName = [x for x in os.listdir(os.path.join(root_path, split)) if x.endswith(".jpg")]
    for img in imgName:
        img_path = os.path.join(root_path, split+"\\"+img)
        data = Data(img_path)
        data.label.classification = Classification(split)
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")




