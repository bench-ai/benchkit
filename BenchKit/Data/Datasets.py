import json
import math
import os
import uuid
import requests
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from typing import final
import shutil
from BenchKit.Data.FileSaver import BaseFile, TextFile, BooleanFile, NumpyFile, JsonFile, TorchFile, NumericFile
from BenchKit.Miscellaneous.User import get_get_url, get_current_dataset, get_ds_chunks


class ProcessorDataset:

    def __init__(self):
        self._prefix = None

    def _get_savers(self) -> tuple[BaseFile, ...]:
        raise NotImplementedError("get_savers must be implemented and must return all savers")

    def prepare(self) -> None:
        for i in self._get_savers():
            i.prefix = self.prefix

    def save_savers(self):
        for i in self._get_savers():
            name, tag = i.save()
            yield name, tag

    def reset_savers(self):
        for i in self._get_savers():
            i.reset()

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str):
        self._prefix = prefix

    def __len__(self):
        raise NotImplementedError("Subclasses of ProcessorDataset should implement __len__.")

    def __iter__(self):
        self._len_list = np.random.shuffle(np.arange(len(self), dtype=int))
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= len(self):
            raise StopIteration

        self._pos += 1
        return self[self._pos - 1]

    @final
    def __getitem__(self, idx) -> tuple[tuple, tuple]:

        return self._get_data(idx)

    def _get_data(self,
                  idx: int) -> tuple:

        raise NotImplementedError("Subclasses of ProcessorDataset should implement _get_label.")


class IterableChunk(IterableDataset):

    def __init__(self,
                 name: str,
                 cloud: bool):

        self._cloud = cloud
        self._name = name
        self.chunk_list = None

        dataset = get_current_dataset(name)

        if not dataset:
            raise RuntimeError("Dataset does not exist")

        self._dataset_id = dataset["id"]
        self.length = dataset["sample_count"]
        self.end_index = dataset["sample_count"]
        self.start_index = None
        self.init_start_index = 0
        self._file_converters = []

    @staticmethod
    def delete_dir(uid: str):
        zip_dir = os.path.join(".", f"Temp-zip-{uid}")
        root_dir = os.path.join(".", uid)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)

    def unzip_local_data(self, current_file: str):

        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join(".", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        os.mkdir(root_dir)
        chunk_path = os.path.join(root_dir,
                                  os.path.split(current_file)[-1][:-len(".tar.gz")])

        shutil.unpack_archive(current_file, chunk_path)

        self.file_converters = chunk_path

        return chunk_path, f_id

    def unzip_cloud_data(self,
                         chunk_id: str,
                         current_file: str):

        download_url = get_get_url(chunk_id)

        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join(".", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        mem_zip = requests.get(download_url)

        zip_dir = os.path.join(".", f"Temp-zip-{f_id}")
        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)

        os.mkdir(zip_dir)
        zip_path = os.path.join(zip_dir, os.path.split(current_file)[-1])

        with open(zip_path, 'wb') as f:
            f.write(mem_zip.content)

        chunk_path = os.path.join(root_dir,
                                  os.path.split(current_file)[-1][:-len(".tar.gz")])

        try:
            shutil.unpack_archive(zip_path, chunk_path)
        except FileExistsError:
            pass

        self.file_converters = chunk_path

        return chunk_path, f_id

    @staticmethod
    def file_converter(tag: str,
                       file_path: str) -> BaseFile:

        match tag:
            case "textfile":
                return TextFile.load(file_path)
            case "bool":
                return BooleanFile.load(file_path)
            case "arr":
                return NumpyFile.load(file_path, False)
            case "enforced_arr":
                return NumpyFile.load(file_path, True)
            case "json":
                return JsonFile.load(file_path)
            case "ten":
                return TorchFile.load(file_path)
            case "num":
                return NumericFile.load(file_path)
            case _:
                raise ModuleNotFoundError(f"tag: {tag} is not a valid tag")

    @property
    def file_converters(self):
        return self._file_converters

    @file_converters.setter
    def file_converters(self, folder_path: str):

        order_list = []
        with open(os.path.join(folder_path, "ann.json"), "r") as f:
            order_list.extend(json.load(f))

        order_list = [
            IterableChunk.file_converter(tag, os.path.join(folder_path, name)) for name, tag in order_list
        ]

        self._file_converters = order_list

    def unpack_data(self,
                    idx: int):

        return *[i(idx) for i in self.file_converters],

    def _data_iterator(self):
        current_count = 0
        previous_count = 0
        current_folder = None

        while self.start_index < self.end_index:

            while current_count <= self.start_index:

                current_file = self.chunk_list.pop(0)
                previous_count = current_count

                if self._cloud:
                    current_count += current_file["file_count"]
                else:
                    current_count += int(os.path.split(current_file)[-1].split("-")[2][:-len(".tar.gz")])

                if current_count > self.start_index:

                    if self._cloud:
                        _, folder = self.unzip_cloud_data(current_file["id"], current_file["location"])
                    else:
                        _, folder = self.unzip_local_data(current_file)

                    if current_folder:
                        self.delete_dir(current_folder)

                    current_folder = folder

            index = self.start_index - previous_count

            yield self.unpack_data(index)

            self.start_index += 1

        self.delete_dir(current_folder)

    def __iter__(self):

        self.start_index = self.init_start_index

        if not self._cloud:
            data_path = f"ProjectDatasets/{self._name}"
            data_list = [i for i in os.listdir(data_path) if i.endswith(".tar.gz")]
            self.chunk_list = [os.path.join(data_path, chunk) for chunk in data_list]
            self.chunk_list = sorted(self.chunk_list, key=lambda x: int(os.path.split(x)[-1].split("-")[1]))
        else:
            self.chunk_list = get_ds_chunks(self._dataset_id)
            self.chunk_list = sorted(self.chunk_list, key=lambda x: x["number"])

        return iter(self._data_iterator())

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        overall_start = dataset.init_start_index
        overall_end = dataset.end_index

        per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))

        dataset.init_start_index = overall_start + worker_id * per_worker
        dataset.end_index = min(dataset.init_start_index + per_worker, overall_end)
