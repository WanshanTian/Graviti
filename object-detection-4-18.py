from tensorbay import GAS
from tensorbay.dataset import Dataset
from tensorbay.label import LabeledBox2D
import os

access_key = "Accesskey-1560922ddbded0491fe0bad4b4b24786"
gas = GAS(access_key)
gas.create_dataset("airplane")
lists = list(gas.list_dataset_names())
path = "F:\\BaiduNetDiskDownload\\dataset"
dateset = Dataset(path)
dataset_client = gas.upload_dataset(dateset,jobs=8,skip_uploaded_files=False)
dataset_client.commit("airplane")


DATASET_NAME = "BSTLD"

_LABEL_FILENAME_DICT = {

    "train": "train.yaml",

}


def BSTLD(path: str) -> Dataset:
    import yaml  # pylint: disable=import-outside-toplevel

    root_path = os.path.abspath(os.path.expanduser(path))

    dataset = Dataset(DATASET_NAME)
    dataset.load_catalog(os.path.join(path, "catalog.json"))

    for mode, label_file_name in _LABEL_FILENAME_DICT.items():
        segment = dataset.create_segment(mode)
        label_file_path = os.path.join(root_path, label_file_name)

        with open(label_file_path, encoding="utf-8") as fp:
            labels = yaml.load(fp, yaml.FullLoader)

        for label in labels:
            if mode == "test":
                # the path in test label file looks like:
                # /absolute/path/to/<image_name>.png
                file_path = os.path.join(root_path, "rgb", "test", label["path"].rsplit("/", 1)[-1])
            else:
                # the path in label file looks like:
                # ./rgb/additional/2015-10-05-10-52-01_bag/<image_name>.png
                file_path = os.path.join(root_path, *label["path"][2:].split("/"))
            data = Data(file_path)
            data.label.box2d = [
                LabeledBox2D(
                    box["x_min"],
                    box["y_min"],
                    box["x_max"],
                    box["y_max"],
                    category=box["label"],
                    attributes={"occluded": box["occluded"]},
                )
                for box in label["boxes"]
            ]
            segment.append(data)

    return dataset
