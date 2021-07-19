from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification
from tensorbay.dataset import Data
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file

config.timeout = 40
config.max_retries = 4

root_path = "E:\\ShannonT\\dL-d采atasets\\vehicle-logos-dataset-master"
dataset_name = "vehicle-logos"

imgs_label = read_csv_file(root_path, "structure.csv", 2, 1, 3)
# print(imgs_label)

classes = []
for img in list(imgs_label.keys()):
    if imgs_label[img][0] not in classes:
        classes.append(imgs_label[img][0])

initial = initial(root_path, dataset_name, ["CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train & test")
for img in list(imgs_label.keys()):
    img_path = os.path.join(os.path.join(root_path, "Images"), img.split("/")[1])
    data = Data(img_path)
    data.label.classification = Classification(imgs_label[img][0])
    segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
