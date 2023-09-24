import json
import math
import os
import shutil
import uuid
import warnings
from typing import final

import numpy as np
import requests
import torch
from torch.utils.data import IterableDataset

from benchkit.data.file_saver import BaseFile
from benchkit.data.file_saver import BooleanFile
from benchkit.data.file_saver import JsonFile
from benchkit.data.file_saver import NumericFile
from benchkit.data.file_saver import NumpyFile
from benchkit.data.file_saver import RawFile
from benchkit.data.file_saver import TextFile
from benchkit.data.file_saver import TorchFile
from benchkit.misc.requests.dataset import get_current_dataset
from benchkit.misc.requests.dataset import get_ds_chunks
from benchkit.misc.requests.dataset import get_get_url


class ProcessorDataset:
    def __init__(self):
        self._prefix = None

    def _get_savers(self) -> tuple[BaseFile, ...] | BaseFile:
        raise NotImplementedError(
            "get_savers must be implemented and must return all savers"
        )

    def _check_savers(self) -> tuple[BaseFile, ...]:
        savers = self._get_savers()
        if not isinstance(savers, tuple):
            if not isinstance(savers, BaseFile):
                raise ValueError("All savers must inherit from BaseFile ")
            else:
                return (savers,)
        else:
            return savers

    def prepare(self) -> None:
        for i in self._check_savers():
            i.prefix = self.prefix

    def save_savers(self):
        for i in self._check_savers():
            name, tag = i.save()
            yield name, tag

    def reset_savers(self):
        for i in self._check_savers():
            i.reset()

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str):
        self._prefix = prefix

    def __len__(self):
        raise NotImplementedError(
            "Subclasses of ProcessorDataset should implement __len__."
        )

    def __iter__(self):
        self._len_list = np.arange(len(self), dtype=int)
        np.random.shuffle(self._len_list)
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= len(self):
            raise StopIteration

        self._pos += 1
        return self[self._len_list[self._pos - 1]]

    @final
    def __getitem__(self, idx):
        self._get_data(idx)

    def _get_data(self, idx: int):
        raise NotImplementedError(
            "Subclasses of ProcessorDataset should implement _get_data."
        )


class IterableChunk(IterableDataset):
    def __init__(self):
        self._cloud = None
        self._name = None
        self.chunk_list = None
        self._dataset_id = None
        self.end_index = None
        self.start_index = None
        self.init_start_index = 0
        self._file_converters = []
        self.length = None

    def post_init(self, name, cloud):
        self._cloud = cloud
        self._name = name

        dataset = get_current_dataset(name)

        if not dataset:
            raise RuntimeError("Dataset does not exist")

        self._dataset_id = dataset["id"]
        self.end_index = dataset["sample_count"]
        self.length = dataset["sample_count"]

    def __len__(self):
        # possibly dangerous
        return self.length

    def test_init(self, name: str, length: int):
        warnings.warn(
            "Warning this method should only be used  for local testing purposes",
            stacklevel=2,
        )
        self._cloud = False
        self._name = name
        self.end_index = length

    @staticmethod
    def delete_dir(uid: str):
        zip_dir = os.path.join("", f"Temp-zip-{uid}")
        root_dir = os.path.join("", uid)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)

    def unzip_local_data(self, current_file: str):
        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join("", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        os.mkdir(root_dir)
        chunk_path = os.path.join(
            root_dir, os.path.split(current_file)[-1][: -len(".tar.gz")]
        )

        shutil.unpack_archive(current_file, chunk_path)

        self.file_converters = chunk_path

        return chunk_path, f_id

    def unzip_cloud_data(self, chunk_id: str, current_file: str):
        download_url = get_get_url(chunk_id)

        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join("", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        # TODO: Consider adding a timeout here. Ignoring bugbear issue for now ...
        mem_zip = requests.get(download_url)  # noqa S113

        zip_dir = os.path.join("", f"Temp-zip-{f_id}")
        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)

        os.mkdir(zip_dir)
        zip_path = os.path.join(zip_dir, os.path.split(current_file)[-1])

        with open(zip_path, "wb") as f:
            f.write(mem_zip.content)

        chunk_path = os.path.join(
            root_dir, os.path.split(current_file)[-1][: -len(".tar.gz")]
        )

        try:
            shutil.unpack_archive(zip_path, chunk_path)
        except FileExistsError:
            pass

        self.file_converters = chunk_path

        return chunk_path, f_id

    def file_converter(self, tag: str, file_path: str) -> BaseFile:
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
            case "folder":
                return RawFile.load(file_path)
            case _:
                raise ModuleNotFoundError(f"tag: {tag} is not a valid tag")

    @property
    def file_converters(self):
        return self._file_converters

    @file_converters.setter
    def file_converters(self, folder_path: str):
        order_list = []
        with open(os.path.join(folder_path, "ann.json")) as f:
            order_list.extend(json.load(f))

        order_list = [
            self.file_converter(tag, os.path.join(folder_path, name))
            for name, tag in order_list
        ]

        self._file_converters = order_list

    def unpack_data(self, idx: int):
        return (
            (*[i(idx) for i in self.file_converters],)
            if len(self._file_converters) > 1
            else self.file_converters[0](idx)
        )

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
                    current_count += int(
                        os.path.split(current_file)[-1].split("-")[2][: -len(".tar.gz")]
                    )

                if current_count > self.start_index:
                    if self._cloud:
                        _, folder = self.unzip_cloud_data(
                            current_file["id"], current_file["location"]
                        )
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
            self.chunk_list = sorted(
                self.chunk_list, key=lambda x: int(os.path.split(x)[-1].split("-")[1])
            )
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

        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_info.num_workers))
        )

        dataset.init_start_index = overall_start + worker_id * per_worker
        dataset.end_index = min(dataset.init_start_index + per_worker, overall_end)
