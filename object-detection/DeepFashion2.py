from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json


root_path = "G:\\download_dataset\\DeepFashion2"
splits = ["train", "validation","json_for_validation"]
# label=os.path.join(root_path,splits[2]+"\\annos"+"\\000001.json")
label = os.path.join(root_path,splits[2]+"\\val_query.json")
with open(label, encoding='utf-8') as f:

    line = f.readline()
    output = json.loads(line)
print(json.dumps(output, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))