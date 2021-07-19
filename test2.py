from common.dataset_initial import INITIAL, update_catalog
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data, Dataset, Segment
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
from tensorbay import GAS
from config import access_key

dataset_name = "MSRA10K-test"
root_path = "G:\\download_dataset\\MSRA10K\\MSRA10K_Imgs_GT\\MSRA10K_Imgs_GT"

# update_catalog(root_path, dataset_name, ["CLASSIFICATION"], ["person"])

gas = GAS(access_key)

# dataset = Dataset(dataset_name)
dataset_client = gas.get_dataset(dataset_name)
draft_number = dataset_client.create_draft("draft_update_label")
segment_client = dataset_client.get_segment("train&test")

i = 1
imgs_fileName = [x for x in os.listdir(os.path.join(root_path, "Imgs")) if x.endswith(".jpg")]
# segment = Segment("train&test", dataset_client)
for img_fileName in imgs_fileName:
    img_path = os.path.join(root_path, "Imgs\\" + img_fileName)
    data = Data(img_path)
    data.label.classification = Classification("person")
    segment_client.upload_label(data)
    i += 1
    print(i)

dataset_client.commit("update labels")
