import os

from tensorbay.client import config
from tensorbay.dataset import Data
from tensorbay.label import Classification

from common.dataset_initial import initial

config.timeout = 40
config.max_retries = 4

root_path = "G:\\shannon\\deeplearning_dataset\\Logo-2K+\\Logo-2K+"
dataset_name = "Logo-2K"

big_class = os.listdir(root_path)
if "catalog.json" in big_class:
    big_class.remove("catalog.json")

classes = []
for i in big_class:
    classes += os.listdir(os.path.join(root_path, i))

initial = initial(root_path, dataset_name, ["CLASSIFICATION"], classes)
gas, dataset = initial.generate_catalog()

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
            data.label.classification = Classification(j)
            segment.append(data)

dataset_client = gas.upload_dataset(dataset, jobs=12)
dataset_client.commit("Initial commit")
