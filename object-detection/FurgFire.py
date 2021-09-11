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

config.timeout = 40
config.max_retries = 4

root_path = "D:\download_dataset\FurgFire\\furg-fire-dataset-master"
dataset_name = "FurgFire"
# videos = [x for x in os.listdir(root_path) if x.endswith(".mp4")]
#
# for videoName in videos:
#     print("now is "+ videoName)
#     folder_name = os.path.join(root_path, videoName.split(".")[0])
#     os.makedirs(folder_name, exist_ok=True)  # 创建与视频同名目录保存截取的图片
#     video_path = os.path.join(root_path, videoName)
#     vc = cv2.VideoCapture(video_path)  # 读入视频文件，打开视频
#     c = 0
#     timeS = 20  # 视频帧计数间隔频率
#     rval = vc.isOpened()  # 视频是否打开成功，返回true表示成功，false表示不成功
#
#     while rval:  # 循环读取视频帧
#         c = c + 1
#         # cap.read()按帧读取视频，ret, frame是获cap.read()方法的两个返回值。
#         # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
#         # frame就是每一帧的图像，是个三维矩阵。
#         rval, frame = vc.read()
#         # print(rval, frame)
#         pic_path = folder_name + '//'  # 图片保存目录
#         if rval:  # c%timeS==0：c除尽timeS时取帧保存，图片，即隔timeS保存一次图片
#             pic_name = videoName.split(".")[0] + '_' + str(c) + '.jpg'
#             cv2.imencode('.jpg', frame)[1].tofile(pic_path + pic_name)
#             # print(pic_path + pic_name)  # 打印生成的路径名
#             cv2.waitKey(1)
#             # cv2.waitKey()参数是1，表示延时1ms切换到下一帧图像，参数过大如cv2.waitKey(1000)，会因为延时过久而卡顿感觉到卡顿。
#             # 参数为0，如cv2.waitKey(0)只显示当前帧图像，相当于视频暂停。
#         else:
#             break
#     vc.release()
splits = [x.split(".")[0] for x in os.listdir(root_path) if x.endswith(".mp4")]
initial = INITIAL(root_path, dataset_name, ["BOX2D"],
                  splits)
gas, dataset = initial.generate_catalog()

for split in splits:
    segment = dataset.create_segment(split)
    imgsName = os.listdir(os.path.join(root_path, split))
    label_path = os.path.join(root_path, split + ".xml")
    # acquire labels
    DOMTree = xml.dom.minidom.parse(label_path)
    collection = DOMTree.documentElement
    frames = collection.getElementsByTagName("frames")
    all = frames[0].getElementsByTagName("_")
    img_lable = []
    for frame in all:
        annotations = frame.getElementsByTagName("annotations")
        # print(annotations)
        if len(annotations) != 0:
            l = []
            for each in annotations:
                ann = each.getElementsByTagName("_")
                if len(ann) != 0:
                    xy = ann[0].childNodes[0].data
                    l.append(xy.split())
            img_lable.append(l)
    for img in imgsName:
        img_path = os.path.join(root_path, split + "\\" + img)
        new_path, num = img.rsplit("_", 1)
        data = Data(img_path, target_remote_path=f"{new_path}_{num.zfill(9)}")
        index = int(img.split(".")[0].split("_")[-1])
        label = img_lable[index]
        if len(label) != 0:
            data.label.box2d = []
            for ii in range(len(label)):
                xmin = float(label[ii][0])
                ymin = float(label[ii][1])
                xmax = float(label[ii][2])
                ymax = float(label[ii][3])
                data.label.box2d.append(LabeledBox2D(xmin, ymin, xmin + xmax, ymin + ymax,
                                                     category="FIRE",
                                                     # attributes={"occluded": box["occluded"]}))
                                                     ))
        segment.append(data)
dataset_client = gas.upload_dataset(dataset, jobs=12, skip_uploaded_files=True)
dataset_client.commit("Initial commit")
