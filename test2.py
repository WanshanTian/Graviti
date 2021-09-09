from common.dataset_initial import INITIAL
from tensorbay.label import LabeledBox2D, Classification, LabeledKeypoints2D
from tensorbay.dataset import Data, Dataset
from tensorbay.geometry import Keypoint2D
from common.label_acquire import acquire_label_xml
import os
from tensorbay.client import config
from common.file_read import read_csv_file
import json
from tensorbay import GAS
from config import access_key
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom

# config.timeout = 40
# config.max_retries = 4
# root_path = "D:\download_dataset\AU_AIR\\auair2019data"
# dataset_name = "AU_AIR"
#
# label_path = "D:\download_dataset\AU_AIR\\auair2019annotations\\annotations.json"
# with open(label_path, encoding='utf-8') as f:
#     line = f.readline()
#     all = json.loads(line)
# classes = all.get("categories")
# print(classes)
# splits = {}
# for i, j in zip(range(8), classes):
#     splits[i] = j
# # print(json.dumps(all, sort_keys=True, indent=4, separators=(', ', ': ')))
# labels = {}
# for instance in all["annotations"]:
#     labels[instance.get("image_name")] = instance.get("bbox")
#     # print(instance)
# print("frame_20190829091111_x_0000234.jpg's bbox is " + str(labels["frame_20190829091111_x_0000234.jpg"]))
# print("frame_20190829091111_x_0000235.jpg's bbox is " + str(labels["frame_20190829091111_x_0000235.jpg"]))
#
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#
# root_path = "D:\\download_dataset\\FSS1000\\fewshot_data"
# classes = sorted(os.listdir(os.path.join(root_path, "fewshot_data")))
# pixel = 1
# for split in classes:
#     imgsName = [x for x in os.listdir(os.path.join(root_path, "fewshot_data\\" + split)) if x.endswith(".png")]
#     print("%s is now transfering , the %d" % (split, pixel))
#     for imgName in imgsName:
#         img_path = os.path.join(root_path, "fewshot_data\\" + split + "\\" + imgName)
#         # img = cv2.imread(img_path, 0).astype("uint16")
#         img=Image.open(img_path,)
#         for i in range(img.shape[0]):
#             for j in range(img.shape[1]):
#                 if img[i, j] != 0:
#                     img[i, j] = pixel
#         img.astype("uint16")
#         png = Image.fromarray(img, mode="I;16")
#         png.save(os.path.join(root_path, "fewshot_data\\" + split + "\\" + imgName.split(".")[0] + "mask.png"))
#         # cv2.imwrite(os.path.join(root_path, "fewshot_data\\" + split + "\\" + imgName.split(".")[0] + "mask.png"), img)
#         # plt.imsave(os.path.join(root_path, "fewshot_data\\" + split + "\\" + imgName.split(".")[0] + "mask.png"), img,
#                    # cmap="gray")
#     pixel += 1
#
# #

# img = Image.open("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\ab_wheel\\1.png")
# a=np.array(img)
# print(a.shape)


# img.astype("uint8")

# cv2.imwrite("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\ab_wheel\\1maask.png", img)
# img = cv2.imread("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\ab_wheel\\1maask.png")

#
# img = cv2.imread("D:\download_dataset\FSS1000\\fewshot_data\\fewshot_data\\abacus\\1mask.png", 0)
# print(img.shape)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if img[i, j] != 0:
#             print(img[i, j])
#             # j = 1
