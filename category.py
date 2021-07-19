from config import *
import os
import json

keypointsDefault = {
    "number": 14,
    "names": ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
              'left_hip',
              'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck'],
    "skeleton": [[10, 8], [8, 6], [11, 9], [9, 7], [6, 7], [0, 6], [1, 7], [0, 1], [0, 2], [1, 3],
                 [2, 4], [3, 5]],
    "visible": "TERNARY"
}

keypointsFace = {
    "number": 20,
    "names": ['right eye pupil',
              "left eye pupil",
              "right_mouth_corner",
              "left mouth corner",
              "outer end of right eye brow",
              "inner end of right eye brow",
              "inner end of left eye brow",
              "outer end of left eye brow",
              "right temple",
              "outer corner of right eye",
              "inner corner of right eye",
              "inner corner of left eye",
              "outer corner of left eye",
              "left temple",
              "tip of nose",
              "right nostril",
              "left nostril",
              "centre point on outer edge of upper lip",
              "centre point on outer edge of lower lip",
              "tip of chin"],
    "skeleton": [[4, 9], [9, 0], [0, 10], [10, 5], [6, 11], [11, 1], [1, 12], [12, 7], [15, 14], [14, 16],
                 [2, 17], [17, 3], [3, 18], [18, 2]],
    "visible": "BINARY"
}


def catalog(tp: list, path: str, args: list, keypoints=None):
    """
    :param tp: BOX2D, CLASSIFICATION, KEYPOINTS2D ...
    :param path: ROOT_PATH
    :param args: CATAGORIES
    :param keypoints: SPECIFIED KEYPOINTS
    :return: NONE
    """
    if keypoints is None:
        keypoints = keypointsDefault
    category = {}
    keypoints = {
        "keypoints": [
            keypoints
        ]
    }
    for i in tp:
        if i == "BOX2D" or i == "CLASSIFICATION" or i == "KEYPOINTS2D":
            category[i] = {"categories": []}
            for j in range(len(args)):
                category[i].get("categories").append({"name": args[j]})
        if i == "BOX2D":
            category[i]["attributes"] = [{"name": "occluded", "type": "boolean", "crowdIndex": "occluded"}]
        if i == "KEYPOINTS2D":
            category[i]["keypoints"] = keypoints["keypoints"]
    with open(os.path.join(path, "catalog.json"), 'w') as file_obj:
        json.dump(category, file_obj)
