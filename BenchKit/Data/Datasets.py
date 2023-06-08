import math
import os
import uuid
import requests
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Any, final
import shutil
from BenchKit.Miscellaneous.User import get_dataset, get_get_url, get_current_dataset


class ProcessorDataset(Dataset):

    def get_label_and_numeric_data(self, item) -> Any:
        pass

    def get_file(self, item) -> list[str] | None:
        return None

    @final
    def __getitem__(self, item) -> tuple[Any, list[str]] | tuple[Any]:
        cur_file = self.get_file(item)
        cur_num = self.get_label_and_numeric_data(item)

        return (cur_num, cur_file) if cur_file else (cur_num,)


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

    @staticmethod
    def delete_dir(uid: str):
        zip_dir = os.path.join(".", f"Temp-zip-{uid}")
        root_dir = os.path.join(".", uid)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)

    @staticmethod
    def _set_new_data(folder_path):
        folder_list = os.listdir(folder_path)

        file_chunk = None
        label_chunk = None
        for i in folder_list:
            path = os.path.join(folder_path, i)
            if os.path.isdir(path):
                file_chunk = sorted(os.listdir(path),
                                    key=lambda x: int(x.split("-")[-1]))

                file_chunk = [os.path.join(path, x) for x in file_chunk]
            else:
                label_chunk = torch.load(path)

        return file_chunk, label_chunk

    def unzip_local_data(self, current_file: str):

        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join(".", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        os.mkdir(root_dir)
        chunk_path = os.path.join(root_dir,
                                  os.path.split(current_file)[-1])

        shutil.unpack_archive(current_file, chunk_path)

        return IterableChunk._set_new_data(chunk_path), f_id

    def unzip_cloud_data(self, current_file: str):
        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join(".", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        file_data = get_get_url(dataset_id=self._dataset_id,
                                file_path=current_file)

        mem_zip = requests.get(file_data)

        zip_dir = os.path.join(".", f"Temp-zip-{f_id}")
        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)

        os.mkdir(zip_dir)
        zip_path = os.path.join(zip_dir, os.path.split(current_file)[-1])

        with open(zip_path, 'wb') as f:
            f.write(mem_zip.content)

        chunk_path = os.path.join(root_dir,
                                  os.path.split(current_file)[-1])

        try:
            shutil.unpack_archive(zip_path, chunk_path)
        except FileExistsError:
            pass

        return IterableChunk._set_new_data(chunk_path), f_id

    @staticmethod
    def get_files(file_folder: str | None) -> list[str] | None:

        if file_folder:
            return [os.path.join(file_folder, i) for i in os.listdir(file_folder)]
        else:
            return None

    def _data_iterator(self):
        current_count = 0
        previous_count = 0
        files = None
        labels = None
        current_folder = None

        while self.start_index < self.end_index:
            while current_count <= self.start_index:
                current_file = self.chunk_list.pop(0)
                previous_count = current_count
                current_count += int(os.path.split(current_file)[-1].split("-")[2])

                if current_count > self.start_index:

                    if self._cloud:
                        files_labels_tuple, folder = self.unzip_cloud_data(current_file)
                    else:
                        files_labels_tuple, folder = self.unzip_local_data(current_file)

                    files, labels = files_labels_tuple

                    if current_folder:
                        self.delete_dir(current_folder)

                    current_folder = folder

            index = self.start_index - previous_count
            yield labels[index], IterableChunk.get_files(files[index]) if files else None
            self.start_index += 1

    def __iter__(self):
        self.start_index = self.init_start_index

        if not self._cloud:
            data_path = f"ProjectDatasets/{self._name}"
            self.chunk_list = [os.path.join(data_path, chunk) for chunk in os.listdir(data_path)]
        else:
            self.chunk_list = get_dataset(self._dataset_id)

        self.chunk_list = sorted(self.chunk_list, key=lambda x: int(os.path.split(x)[-1].split("-")[1]))

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
