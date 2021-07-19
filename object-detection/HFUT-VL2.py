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
dataset_name = "HFUT-VL2-dataset"

category = ["HFUT-VL1", "HFUT-VL2"]

sub_category_1 = ["HFUT-VL1_1", "HFUT-VL1_2"]
sub_category_2 = ["HFUT-VL2_1", "HFUT-VL2_2"]

classes = []
classes_2 = {}
with open("E:\\ShannonT\\dL-datasets\\HFUT-VL-dataset-master\\HFUT-VL2\\CarNames.txt", "r") as f:
    for line in f.readlines():
        l = line.strip("\n")
        classes_2[l.split()[1]] = l.split()[0]
        classes.append(l.split()[0])

initial = initial(root_path, dataset_name, ["CLASSIFICATION", "BOX2D"], classes)
gas, dataset = initial.generate_catalog()

segment = dataset.create_segment("train & test")
imgs_file = os.listdir(os.path.join(os.path.join(root_path, category[1]), sub_category_2[0])) + os.listdir(
    os.path.join(os.path.join(root_path, category[1]), sub_category_2[1]))

for img in imgs_file:
    if int(img.split("_")[0]) < 41:
        path = os.path.join(os.path.join(os.path.join(root_path, category[1]), sub_category_2[0]), img)
    else:
        path = os.path.join(os.path.join(os.path.join(root_path, category[1]), sub_category_2[1]), img)
    cate = classes_2[img.split("_")[0]]
    data = Data(path)
    img_label = acquire_label_xml(
        os.path.join("E:\\ShannonT\\dL-datasets\\HFUT-VL-dataset-master\\HFUT-VL2\\HFUT-VL2-annotation",
                     img.split(".")[0] + ".xml"))
    data.label.classification = Classification(cate)
    if len(img_label) != 0:
        data.label.box2d = []
        for i in range(len(img_label)):
            xmin = img_label[i][0]
            ymin = img_label[i][1]
            xmax = img_label[i][2]
            ymax = img_label[i][3]
            data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                 category=img_label[i][4],
                                                 # attributes={"occluded": box["occluded"]}))
                                                 ))
    segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
