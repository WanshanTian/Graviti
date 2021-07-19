from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification
from tensorbay.dataset import Data
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file

config.timeout = 40
config.max_retries = 4

root_path = "E:\\ShannonT\\dL-datasets\\car-logos-dataset-master\\logos"
dataset_name = "Car-Logos-Dataset"

files = os.listdir(os.path.join(root_path, "original"))
classes = []
for img in files:
    path = os.path.join(os.path.join(root_path, "original"), img)
    classes.append(img.split(".")[0])

initial = initial(root_path, dataset_name, ["CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

for i in ["optimized", "original", "thumb"]:
    files = os.listdir(os.path.join(root_path, i))
    classes = []
    for img in files:
        path = os.path.join(os.path.join(root_path, i), img)
        classes.append(img.split(".")[0])

    segment = dataset.create_segment(i)

    for img in files:
        cate = img.split(".")[0]
        path = os.path.join(os.path.join(root_path, i), img)
        data = Data(path)
        data.label.classification = Classification(cate)
        segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
