from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from config import *
from category import catalog
from tensorbay.label import LabeledBox2D
import os
from xml.dom.minidom import parse
import xml.dom.minidom

# configuration
root_path = "G:\\shannon\\deeplearning_dataset\\swimming-pool-and-car-detection"
dataset_name = "swimming pool and car detection"

# generate catalog
if "catalog.json" not in os.listdir(root_path):
    catalog(root_path, "pool", "car")

# create dataset
gas = GAS(access_key)
if dataset_name not in list(gas.list_dataset_names()):
    gas.create_dataset(dataset_name)

# load catalog
dataset = Dataset(dataset_name)
dataset.load_catalog(os.path.join(root_path, "catalog.json"))


for k in ["training_data", "test_data_images"]:
    if k == "training_data":
        segment = dataset.create_segment("train")
        imgs_name = os.listdir(os.path.join(root_path, "training_data\\training_data\\images"))
        image_files_path = [os.path.join(os.path.join(root_path, "training_data\\training_data\\images"), i) for i in
                            imgs_name]
        # acquire bounding box
        imgs_lable_dict = {}
        for img_name in imgs_name:
            DOMTree = xml.dom.minidom.parse(
                os.path.join(os.path.join(root_path, "training_data\\training_data\\labels"),
                             img_name.split(".")[0] + ".xml"))
            collection = DOMTree.documentElement
            boundingbox = collection.getElementsByTagName("object")
            imgs_lable_dict[img_name] = []
            for i in boundingbox:
                category = i.getElementsByTagName("name")[0].childNodes[0].data
                tmp=[]
                tmp.append(float(
                    [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")][
                        0]))
                tmp.append(float(
                    [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")][
                        0]))
                tmp.append(float(
                    [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")][
                        0]))
                tmp.append(float(
                    [j.childNodes[0].data for j in i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")][
                        0]))
                tmp.append(category)
                imgs_lable_dict[img_name].append(tmp)


            data = Data(os.path.join(os.path.join(root_path, "training_data\\training_data\\images"), img_name))
            if img_name in imgs_lable_dict.keys():
                labels = imgs_lable_dict[img_name]
            else:
                labels = []
            if len(labels) != 0:
                data.label.box2d = []
                for i in range(len(labels)):
                    if labels[i][4] == "1":
                        cate = "car"
                    if labels[i][4] == "2":
                        cate = "pool"
                    xmin = labels[i][0]
                    ymin = labels[i][1]
                    xmax = labels[i][2]
                    ymax = labels[i][3]
                    data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                         category=cate,
                                                         # attributes={"occluded": box["occluded"]}))
                                                         ))
            segment.append(data)

    if k == "test_data_images":
        segment = dataset.create_segment("test")
        imgs_name = os.listdir(os.path.join(root_path, "test_data_images\\test_data_images\\images"))
        image_files_path = [os.path.join(os.path.join(root_path, "test_data_images\\test_data_images\\images"), i) for i
                            in
                            imgs_name]
        for img_name in imgs_name:
            data = Data(os.path.join(os.path.join(root_path, "test_data_images\\test_data_images\\images"), img_name))
            segment.append(data)


dataset_client = gas.upload_dataset(dataset)
dataset_client.commit("Initial commit")
