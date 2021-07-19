from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification
from tensorbay.dataset import Data
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file

config.timeout = 40
config.max_retries = 4

root_path = "E:\\ShannonT\\dL-datasets\\HFUT-VL-dataset-master"
dataset_name = "HFUT-VL1-dataset"

category = ["HFUT-VL1", "HFUT-VL2"]

sub_category_1 = ["HFUT-VL1_1", "HFUT-VL1_2"]
sub_category_2 = ["HFUT-VL2_1", "HFUT-VL2_2"]

classes = []
classes_1 = {}
with open("E:\\ShannonT\\dL-datasets\\HFUT-VL-dataset-master\\HFUT-VL1\\CarNames.txt", "r") as f:
    for line in f.readlines():
        l = line.strip("\n")
        classes_1[l.split()[1]] = l.split()[0]
        classes.append(l.split()[0])
classes_2 = {}
with open("E:\\ShannonT\\dL-datasets\\HFUT-VL-dataset-master\\HFUT-VL2\\CarNames.txt", "r") as f:
    for line in f.readlines():
        l = line.strip("\n")
        classes_2[l.split()[1]] = l.split()[0]

initial = initial(root_path, dataset_name, ["CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train & test")

imgs_file = os.listdir(os.path.join(os.path.join(root_path, category[0]), sub_category_1[0])) + os.listdir(
    os.path.join(os.path.join(root_path, category[0]), sub_category_1[1]))

i = 0
for img in imgs_file:
    if i < 8000:
        path = os.path.join(os.path.join(os.path.join(root_path, category[0]), sub_category_1[0]), img)
    else:
        path = os.path.join(os.path.join(os.path.join(root_path, category[0]), sub_category_1[1]), img)
    i += 1
    cate = classes_1[img.split("_")[0]]
    data = Data(path)
    data.label.classification = Classification(cate)
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
