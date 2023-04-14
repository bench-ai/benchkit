import json
import os
import requests
import torch
from torch.utils.data import Dataset
from typing import Any, final
import shutil
import multiprocessing

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


# Add the ability to use accelerate
# Change code so it works using multiple gpu's
# -- This means that locks and events have to be gpu specific
# -- make a new lock class
# Also you need to change it so that the dataset size is in the zip name
# -- This allows any process to continue from any dataset slice
# after this make an accelerated model and see if this works

class ChunkDataset(Dataset):

    @staticmethod
    def update_process_count(process_name: str | None = None) -> int:
        with open("Process.json", "r") as j:
            process_dict = json.load(j)

            if process_name:
                process_dict["process_list"].append(process_name)

        with open("Process.json", "w") as j:
            json.dump(process_dict, j)

        # lock.release()
        return len(process_dict["process_list"])

    @staticmethod
    def new_process_json():
        with open("Process.json", "w") as file:
            json.dump({"process_list": []}, file)

    # here add gpu count
    # add num workers per gpu this number should depend on the gpu the user plans to use
    def __init__(self,
                 name: str,
                 cloud: bool):

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
            print(self._chunk_list)
        else:
            self._chunk_list = get_dataset(dataset_id)

        self._dataset_length = dataset_len

        self._pos = len(self._label_chunk)
        self._prev_doc_len = 0
        self._write_file_event = multiprocessing.Event()

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
            lock.acquire()
            p_len = ChunkDataset.update_process_count(str(os.getpid()))
            self._prev_doc_len += len(self._label_chunk)
            c1, c2 = self.set_new_chunk_path()

            if p_len == 4:
                self.unzip_labels_and_files(c1, c2)
                ChunkDataset.new_process_json()
                self._write_file_event.set()
                lock.release()
                self._write_file_event.clear()
            else:
                lock.release()
                self._write_file_event.wait()

            self._set_new_data()
            self._pos += len(self._label_chunk)

        file_tup = self.get_files(idx - self._prev_doc_len)
        if file_tup:
            return self._label_chunk[idx - self._prev_doc_len], self.get_files(idx - self._prev_doc_len)
        else:
            return self._label_chunk[idx - self._prev_doc_len]
