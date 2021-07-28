import os

from tensorbay.client import config
from tensorbay.dataset import Data
from tensorbay.label import Classification

from common.dataset_initial import INITIAL

root_path = "G:\download_dataset\WeedDetectionInSoybeanCrops\weed-detection-in-soybean-crops\dataset"
config.timeout = 400
config.max_retries = 10

dataset_name = "WeedDetectionInSoybeanCrops-test"

classes = os.listdir(root_path)
if "catalog.json" in classes:
    classes.remove("catalog.json")

import cv2


for split in classes:
    imgName = [x for x in os.listdir(os.path.join(root_path, split)) if x.endswith(".tif")]
    for img in imgName:
        img_path = os.path.join(root_path, split+"\\"+img)
        i = cv2.imread(img_path)
        cv2.imwrite(os.path.join(root_path, split+"\\"+img.split(".")[0]+".jpg"), i)
