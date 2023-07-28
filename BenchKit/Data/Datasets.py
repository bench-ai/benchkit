import json
import math
import os
import uuid
import requests
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from typing import Any, final
import shutil
import pickle
from BenchKit.Data.FileSaver import BaseFile
from BenchKit.Miscellaneous.User import get_get_url, get_current_dataset, get_ds_chunks


class ProcessorDataset:

    def __init__(self):
        self.prefix = None

    def get_savers(self) -> tuple[BaseFile, ...]:
        raise NotImplementedError("get_savers must be implemented and must return all savers")

    def prepare(self) -> None:
        for i in self.get_savers():
            i.prefix = self.prefix

    @property
    def prefix(self):
        return self.prefix

    @prefix.setter
    def prefix(self, prefix: str):
        self.prefix = prefix

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

        return self._get_input(idx), self._get_label(idx)

    @staticmethod
    def list_all_converters(curr_class):

        method_list = []

        for att in dir(curr_class):
            if callable(getattr(curr_class, att)) and not att.startswith('__') and att.startswith('_'):
                if att != "_validate":
                    method_list.append(att)

        return method_list

    @staticmethod
    def check_instance(value: Any,
                       enforced_type: tuple):

        if not isinstance(value, enforced_type):
            raise Exception(f"Value: {value}, must be of type {enforced_type} not {type(value)}")

    def validate(self,
                 data: Any,
                 prefix: str,
                 dtype: str | None = None):

        if not dtype:
            if isinstance(data, bool):
                return ProcessorDataset._bool(prefix,
                                              data,
                                              str(uuid.uuid4()))
            elif isinstance(data, int):
                return ProcessorDataset._num(prefix,
                                             data,
                                             str(uuid.uuid4()))
            elif isinstance(data, float):
                return ProcessorDataset._num(prefix,
                                             data,
                                             str(uuid.uuid4()))
            elif isinstance(data, str):

                if os.path.isfile(data):
                    return ProcessorDataset._file(prefix, data, str(uuid.uuid4()))
                else:
                    return ProcessorDataset._txt(prefix, data, str(uuid.uuid4()))

            else:
                raise RuntimeError(f"Cannot infer datatype of {data} please provide annotation")
        else:

            dtype = dtype.lower()

            if not dtype.startswith("_"):
                dtype = f"_{dtype}"

            if dtype in self._converter:
                method = getattr(self, dtype)
                return method(prefix,
                              data,
                              str(uuid.uuid4()))
            else:
                raise RuntimeError(f"{dtype} not a valid method")

    def _get_label(self,
                   idx: int) -> tuple:

        raise NotImplementedError("Subclasses of ProcessorDataset should implement _get_label.")

    def _get_input(self,
                   idx: int) -> tuple:

        raise NotImplementedError("Subclasses of ProcessorDataset should implement _get_input.")

    @staticmethod
    def _file(path_prefix,
              f_path: str,
              name):

        if not os.path.isfile(f_path):
            raise RuntimeError(f"{f_path} does not exist")

        f_name, suffix = os.path.split(f_path)[-1].split(".")

        save_path = f"{path_prefix}/{f_name}-{name}_file.{suffix}"

        shutil.copyfile(f_path, save_path)

        return f_path

    @staticmethod
    def _npy(path_prefix: str,
             np_arr: np.ndarray,
             name: str):

        p = f'{path_prefix}/bench-{name}_npy.npy'

        np.save(p,
                np_arr)

        return p

    @staticmethod
    def _txt(path_prefix: str,
             txt: str,
             name: str,
             dtype: str | None = "txt"):

        p = f'{path_prefix}/bench-{name}_{dtype}.txt'

        ProcessorDataset.check_instance(txt, (str,))
        with open(p, "w") as f:
            f.write(txt)

        return p

    @staticmethod
    def _ten(path_prefix: str,
             tensor: torch.Tensor,
             name: str):

        p = f'{path_prefix}/bench-{name}_ten.pt'

        ProcessorDataset.check_instance(tensor, (torch.Tensor,))
        torch.save(tensor, p)

        return p

    @staticmethod
    def _json(path_prefix: str,
              json_obj: list | dict,
              name: str):

        p = f'{path_prefix}/bench-{name}_json.json'

        ProcessorDataset.check_instance(json_obj, (list, dict))

        with open(p, "w") as f:
            json.dump(json_obj, f)

        return p

    @staticmethod
    def _num(path_prefix: str,
             num: int | float,
             name: str):

        ProcessorDataset.check_instance(num, (int,
                                              float))

        file_str = ""

        if isinstance(num, int):
            with open(f'{path_prefix}/bench-{name}_int.bin', 'wb') as f:
                num_bytes = num.to_bytes((num.bit_length() + 7) // 8, "big")
                f.write(num_bytes)

            file_str = f'{path_prefix}/bench-{name}_int.bin'

        elif isinstance(num, float):
            with open(f'{path_prefix}/bench-{name}_flo.pkl', 'wb') as f:
                pickle.dump(num, f)

            file_str = f'{path_prefix}/bench-{name}_flo.pkl'

        return file_str

    @staticmethod
    def _bool(path_prefix: str,
              boolean: bool,
              name: str):

        ProcessorDataset.check_instance(boolean, (bool,))

        bool_num = int(boolean)
        return ProcessorDataset._txt(path_prefix,
                                     str(bool_num),
                                     name,
                                     dtype="bool")

    @staticmethod
    def _gen(path_prefix: str,
             data: Any,
             name: str):

        p = f'{path_prefix}/bench-{name}_torch_gen.pt'

        torch.save(data, p)

        return p


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

        folder_list = [os.path.join(folder_path, i) for i in os.listdir(folder_path)]

        np.random.shuffle(folder_list)

        return folder_list

    def unzip_local_data(self, current_file: str):

        f_id = f"Temp-{str(uuid.uuid4())}"
        root_dir = os.path.join(".", f_id)

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        os.mkdir(root_dir)
        chunk_path = os.path.join(root_dir,
                                  os.path.split(current_file)[-1][:-len(".tar.gz")])

        shutil.unpack_archive(current_file, chunk_path)

        return IterableChunk._set_new_data(chunk_path), f_id

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

    @staticmethod
    def _gen(folder_name: str,
             file_name: str) -> torch.Tensor:

        return torch.load(os.path.join(folder_name, file_name))

    @staticmethod
    def _bool(folder_name: str,
              file_name: str) -> bool:

        bool_str = IterableChunk._txt(folder_name, file_name)

        return bool(int(bool_str))

    @staticmethod
    def _txt(folder_name: str,
             file_name: str) -> str:

        with open(os.path.join(folder_name, file_name), "r") as f:
            ret_str = f.read()

        return ret_str

    @staticmethod
    def _flo(folder_name: str,
             file_name: str) -> float:

        with open(os.path.join(folder_name, file_name), "rb") as file:
            data = pickle.load(file)
            return data

    @staticmethod
    def _int(folder_name: str,
             file_name: str) -> int:

        with open(os.path.join(folder_name, file_name), "rb") as file:
            binary_data = file.read()
            return int.from_bytes(binary_data, byteorder="big", signed=True)

    @staticmethod
    def _json(folder_name: str,
              file_name: str) -> dict | list:

        with open(os.path.join(folder_name, file_name), "r") as f:
            return json.load(f)

    @staticmethod
    def _file(folder_name: str,
              file_name: str) -> str:

        return os.path.join(folder_name, file_name)

    @staticmethod
    def _npy(folder_name: str,
             file_name: str):

        return np.load(os.path.join(folder_name, file_name))

    def converter(self,
                  folder_name: str,
                  file_name: str):

        name, _ = file_name.split(".")

        method_name = ""
        for i in name[::-1]:
            method_name = i + method_name

            if i == "_":
                break

        if method_name == "_raw":
            return IterableChunk._txt(folder_name, file_name)
        else:
            method = getattr(self, method_name)
            return method(folder_name,
                          file_name)

    def unpack_data(self,
                    folder_path: str):

        j_dict = {}
        with open(os.path.join(folder_path, "ann.json"), "r") as f:
            j_dict.update(json.load(f))

        return {
            "input_list": [self.converter(folder_path, i) for i in j_dict["input_list"]],
            "label_list": [self.converter(folder_path, i) for i in j_dict["label_list"]]
        }

    def _data_iterator(self):
        current_count = 0
        previous_count = 0
        file_list = None
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
                        file_list, folder = self.unzip_cloud_data(current_file["id"], current_file["location"])
                    else:
                        file_list, folder = self.unzip_local_data(current_file)

                    if current_folder:
                        self.delete_dir(current_folder)

                    current_folder = folder

            index = self.start_index - previous_count

            yield self.unpack_data(file_list[index])

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
