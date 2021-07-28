import os
root_path = "G:\download_dataset\SCUT_FBP5500\SCUT-FBP5500_v2.1\SCUT-FBP5500_v2"
img = "AF1.jpg"
img_label_path = os.path.join(root_path, "landmark_txt\\"+img.split(".")[0]+".txt")
labels_origin = []
with open(img_label_path, "r") as f:
    for line in f.readlines():
        l = " ".join(line.split("\t"))
        labels_origin.append(l.strip("\n"))
label = [x.split(" ") for x in labels_origin]
print(labels_origin)