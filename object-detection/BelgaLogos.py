from common.dataset_initial import initial
from tensorbay.label import LabeledBox2D
from tensorbay.dataset import Dataset, Data
import os

root_path = "G:\\shannon\\deeplearning_dataset\\BelgaLogos"
dataset_name = "BelgaLogos"
imgs_name = os.listdir(os.path.join(root_path, "images"))

label_path = os.path.join(root_path, "labels.txt")
labels_origin = []
with open(label_path, "r") as f:
    for line in f.readlines():
        l = " ".join(line.split("\t"))
        labels_origin.append(l.strip("\n"))
imgs_label = {}
classes = []
for i in labels_origin:
    if i.split()[2] not in imgs_label.keys():
        imgs_label[i.split()[2]] = []
    if i.split()[1] not in classes:
        classes.append(i.split()[1])
    imgs_label[i.split()[2]].append(
        [int(i.split()[5]), int(i.split()[6]), int(i.split()[7]), int(i.split()[8]), i.split()[1]])

initial = initial(root_path, dataset_name, "box2d", classes)
gas, dataset = initial.generate_catalog()


for i in ["labelled", "not labelled"]:
    if i == "labelled":
        segment = dataset.create_segment(i)
        for j in list(imgs_label.keys()):
            path = os.path.join(os.path.join(root_path, "images"), j)
            data = Data(path)
            if j not in imgs_label.keys():
                img_label = []
            else:
                img_label = imgs_label[j]
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
    elif i == "not labelled":
        segment = dataset.create_segment(i)
        test_imgs = []
        for ii in imgs_name:
            if ii not in list(imgs_label.keys()):
                test_imgs.append(ii)
        for j in test_imgs:
            path = os.path.join(os.path.join(root_path, "images"), j)
            data = Data(path)
            segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")