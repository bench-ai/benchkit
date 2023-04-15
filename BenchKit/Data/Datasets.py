import json
import os
import requests
import torch
from torch.utils.data import Dataset
from typing import Any, final
import shutil
import multiprocessing
from accelerate import Accelerator
from BenchKit.Miscellaneous.Settings import get_config
from BenchKit.Miscellaneous.User import get_dataset, get_get_url

lock = multiprocessing.Lock()


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


class ChunkDataset(Dataset):

    @staticmethod
    def update_process_count(process_name: str,
                             gpu_process: int) -> int:

        lock.acquire()
        gpu_process = str(gpu_process)
        with open("Process.json", "r") as j:
            process_dict = json.load(j)

            process_list = process_dict.get(gpu_process)

            if process_list:
                process_list.append(process_name)
            else:
                process_dict[gpu_process] = [process_name]

        with open("Process.json", "w") as j:
            json.dump(process_dict, j)

        lock.release()
        return len(process_dict[gpu_process])

    @staticmethod
    def new_process_json():
        lock.acquire()
        with open("Process.json", "w") as file:
            json.dump({}, file)
        lock.release()

    @staticmethod
    def reset_process_list(gpu_process: int):
        lock.acquire()
        gpu_process = str(gpu_process)

        with open("Process.json", "r") as file:
            json_dict = json.load(file)

        json_dict[gpu_process] = []

        with open("Process.json", "w") as j:
            json.dump(json_dict, j)

        lock.release()

    def __init__(self,
                 name: str,
                 cloud: bool,
                 acc: Accelerator,
                 num_workers: int):

        self._accelerator = acc
        self._num_gpu_process = self._accelerator.num_processes
        self._num_workers = num_workers

        cfg = get_config()

        ChunkDataset.new_process_json()

        for i in cfg.get("datasets"):
            if i["name"] == name:
                chunk_path = i["path"]
                dataset_len = i["length"]
                dataset_id = i["info"]["id"]

        self._label_chunk: list = []
        self._file_chunk = None
        self._chunk_path = None

        self._cloud = cloud
        self._dataset_id = dataset_id

        if not cloud:
            self._chunk_list = [os.path.join(chunk_path, chunk) for chunk in os.listdir(chunk_path)]
        else:
            self._chunk_list = get_dataset(dataset_id)

        self._chunk_list = sorted(self._chunk_list, key=lambda x: int(os.path.split(x)[-1].split("-")[1]))
        self._dataset_length = dataset_len

        self._pos = len(self._label_chunk)
        self._prev_doc_len = 0

        self._event_dict = {str(i): multiprocessing.Event() for i in range(self._num_gpu_process)}

    def unzip_labels_and_files(self,
                               current_chunk: str,
                               chunk_path: str):

        root_dir = os.path.join(".", f"TempData")

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        os.mkdir(root_dir)

        if not self._cloud:

            try:
                shutil.unpack_archive(current_chunk, chunk_path)
            except FileExistsError:
                pass
        else:

            file_data = get_get_url(dataset_id=self._dataset_id,
                                    file_path=current_chunk)

            mem_zip = requests.get(file_data)

            zip_dir = os.path.join(".", "TempData-zip")
            if os.path.isdir(zip_dir):
                shutil.rmtree(zip_dir)

            os.mkdir(zip_dir)
            zip_path = os.path.join(zip_dir, os.path.split(current_chunk)[-1])

            with open(zip_path, 'wb') as f:
                f.write(mem_zip.content)

            try:
                shutil.unpack_archive(zip_path, chunk_path)
            except FileExistsError:
                pass

    def set_new_chunk_path(self):
        root_dir = os.path.join(".", f"TempData")
        current_chunk = self._chunk_list.pop(0)
        self._chunk_path = os.path.join(root_dir,
                                        os.path.split(current_chunk)[-1])

        return current_chunk, self._chunk_path

    def _set_new_data(self):
        folder_list = os.listdir(self._chunk_path)

        for i in folder_list:
            path = os.path.join(self._chunk_path, i)
            if os.path.isdir(path):
                self._file_chunk = sorted(os.listdir(path),
                                          key=lambda x: int(x.split("-")[-1]))

                self._file_chunk = [os.path.join(path, x) for x in self._file_chunk]
            else:
                self._label_chunk = torch.load(path)

    def __len__(self):
        return self._dataset_length

    def get_files(self,
                  idx: int) -> list[str] | None:

        if self._file_chunk:
            return [os.path.join(self._file_chunk[idx], i) for i in os.listdir(self._file_chunk[idx])]
        else:
            return None

    def __getitem__(self, idx):
        if idx >= self._pos:
            p_len = ChunkDataset.update_process_count(str(os.getpid()),
                                                      self._accelerator.process_index)
            self._prev_doc_len += len(self._label_chunk)

            while idx >= self._pos:
                c1, c2 = self.set_new_chunk_path()
                curr_len = int(os.path.split(c1)[-1].split("-")[2])
                self._pos += curr_len

            if p_len == self._num_workers:
                self.unzip_labels_and_files(c1, c2)
                ChunkDataset.reset_process_list(self._accelerator.process_index)
                self._event_dict[str(self._accelerator.process_index)].set()
                self._event_dict[str(self._accelerator.process_index)].clear()
            else:
                self._event_dict[str(self._accelerator.process_index)].wait()

            self._set_new_data()

        file_tup = self.get_files(idx - self._prev_doc_len)
        if file_tup:
            return self._label_chunk[idx - self._prev_doc_len], self.get_files(idx - self._prev_doc_len)
        else:
            return self._label_chunk[idx - self._prev_doc_len]
