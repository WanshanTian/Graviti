from tensorbay import GAS
from tensorbay.dataset import Dataset, Data
from category import catalog
from config import *
import datetime
import os


def acquire_dataset(gas, dataset_name):
    if dataset_name not in list(gas.list_dataset_names()):
        dataset_client = gas.create_dataset(dataset_name)
        return dataset_client
    dataset_client = gas.get_dataset(dataset_name)
    return dataset_client


def update_labels(gas, dataset_name, catalog):
    dataset_client = gas.get_dataset(dataset_name)
    dataset_client.create_draft("draft-" + str(datetime.datetime.now()))
    dataset_client.upload_catalog(catalog)
    for segment in dataset_client:
        segment_client = dataset_client.get_segment(segment.name)
        for data in segment_client:
            segment_client.upload_label(data)
            dataset_client.commit("update labels")


class INITIAL():
    """
    generate catalog and upload catalog
    """

    def __init__(self, root_path: str, dataset_name: str, tp: list, args: list):
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.tp = tp
        self.args = args

    def generate_catalog(self, keypoints=None):

        if "catalog.json" not in os.listdir(self.root_path):
            catalog(self.tp, self.root_path, self.args, keypoints)
        else:
            os.remove(os.path.join(self.root_path, "catalog.json"))
            catalog(self.tp, self.root_path, self.args, keypoints)

        # create dataset
        gas = GAS(access_key)
        if self.dataset_name not in list(gas.list_dataset_names()):
            gas.create_dataset(self.dataset_name)

        # load catalog
        dataset = Dataset(self.dataset_name)
        dataset.load_catalog(os.path.join(self.root_path, "catalog.json"))
        return gas, dataset


def update_catalog(root_path, dataset_name, tp: list, args: list):
    initial = INITIAL(root_path, dataset_name, tp, args)
    gas, dataset = initial.generate_catalog()
    dataset_client = gas.upload_dataset(dataset, jobs=12)
    dataset_client.commit("try commit")


# def update_label(dataset_name, imgs_path: list):
#     gas = GAS(access_key)
#     dataset_client = gas.get_dataset(dataset_name)
#     dataset_client.create_draft("draft_update_label")
#     segment_client = dataset_client.get_segment("train&test")
#     for img_path in imgs_path:
#         data = Data(img_path)
#         data.label.classification = Classification("person")
#         segment_client.upload_label(data)
#
#     dataset_client.commit("update labels")
