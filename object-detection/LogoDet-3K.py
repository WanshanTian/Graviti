from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D, Classification
from tensorbay.dataset import Data
from common.label_acquire import acquire_label_xml
import os

root_path = "G:\\shannon\\deeplearning_dataset\\LogoDet-3K"
dataset_name = "LogoDet-3K"

big_class = os.listdir(root_path)
if "catalog.json" in big_class:
    big_class.remove("catalog.json")

classes = []

for i in big_class:
    classes += os.listdir(os.path.join(root_path, i))

initial = initial(root_path, dataset_name, ["BOX2D", "CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

# img_path = "G:\\shannon\\deeplearning_dataset\\LogoDet-3K\\Clothes\\2xist\\1.jpg"
# label_path = "G:\\shannon\\deeplearning_dataset\\LogoDet-3K\\Clothes\\2xist\\1.xml"

for i in big_class:
    subclass = os.listdir(os.path.join(root_path, i))
    segment = dataset.create_segment(i)
    for j in subclass:

        files = os.listdir(os.path.join(os.path.join(root_path, i), j))
        imgs_path = []
        for k in files:
            if k.split(".")[1] == "jpg":
                imgs_path.append(k)
        for img_path in imgs_path:
            img = os.path.join(os.path.join(os.path.join(root_path, i), j), img_path)
            data = Data(img,target_remote_path=j+img_path)
            img_label = acquire_label_xml(os.path.join(os.path.join(os.path.join(root_path, i), j),img_path.split(".")[0]+".xml"))
            data.label.box2d = []
            for ii in range(len(img_label)):
                xmin = img_label[ii][0]
                ymin = img_label[ii][1]
                xmax = img_label[ii][2]
                ymax = img_label[ii][3]
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmax, ymax,
                                                     category=img_label[ii][4],
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
            data.label.classification = Classification(j)
            segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=24)
dataset_client.commit("Initial commit")
