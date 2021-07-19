# with open(os.path.join(root_path, "splits.json"), encoding='utf-8') as f:
#     line = f.readline()
#     all = json.loads(line)
#
#
# label_path = os.path.join(root_path, "labels.txt")
# labels_origin = []
# with open(label_path, "r") as f:
#     for line in f.readlines():
#         l = " ".join(line.split("\t"))
#         labels_origin.append(l.strip("\n"))
#
import os
import csv


def read_csv_file(root_path, csv_file_name, key_index, *value_indexs, ignore_firstline=1):
    imgs_lable_dict = {}
    with open(os.path.join(root_path, csv_file_name), "r") as file:
        csv_file = csv.reader(file)
        if ignore_firstline == 1:
            csv_file.__next__()
        for row in csv_file:
            if row[key_index] not in imgs_lable_dict.keys():
                imgs_lable_dict[row[key_index]] = []
            for value_index in value_indexs:
                imgs_lable_dict[row[key_index]].append(row[value_index])
    return imgs_lable_dict
