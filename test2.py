import json
import os

root_path = "G:\\download_dataset\\StanfordExtra"
label_path = os.path.join(root_path, "StanfordExtra_v1.json")

with open(label_path, encoding='utf-8') as f:
    line = f.readline()
    all = json.loads(line)

img_labels = {}
for img in all:
    img_labels[img.get("img_path").split("/")[-1]] = [img.get("img_bbox"), img.get("joints"),
                                                      img.get("img_path").split("/")[0].split("-")[1]]

# print(img_labels)
print(img_labels.get("n02095314_1814.jpg")[0])
for keypoint in range(int(len(img_labels["n02095314_1814.jpg"][1]))):
    x, y, v = img_labels["n02095314_1814.jpg"][1][keypoint][0], img_labels["n02095314_1814.jpg"][1][keypoint][1], int(
        img_labels["n02095314_1814.jpg"][1][keypoint][2])
    print(x,y,v)
