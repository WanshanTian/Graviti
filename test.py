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
a=1
b="2"
print(f"{a}_{b.zfill(8)}")
print("1".zfill(4))