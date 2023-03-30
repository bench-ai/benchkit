import os
import torch
from torch.utils.data import Dataset
from typing import Any, final
import shutil
import multiprocessing

from BenchKit.Miscellaneous.Settings import get_config


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

    def __init__(self,
                 name: str):

        cfg = get_config()

        for i in cfg.get("datasets"):
            if i["name"] == name:
                chunk_path = i["path"]
                dataset_len = i["length"]

        self._label_chunk: list = []
        self._file_chunk = None
        self._chunk_path = None

        self._chunk_list = [os.path.join(chunk_path, chunk) for chunk in os.listdir(chunk_path)]
        self._dataset_length = dataset_len

        self._pos = len(self._label_chunk)
        self._prev_doc_len = 0

    def get_current_labels_and_files(self):

        process = multiprocessing.current_process().name

        root_dir = os.path.join(".", f"TempData-{process}")

        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)

        os.mkdir(root_dir)

        current_chunk = self._chunk_list.pop(0)
        self._chunk_path = os.path.join(root_dir,
                                        os.path.split(current_chunk)[-1])

        try:
            shutil.unpack_archive(current_chunk, self._chunk_path)
        except FileExistsError:
            pass

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
            self._prev_doc_len += len(self._label_chunk)
            self.get_current_labels_and_files()
            self._pos += len(self._label_chunk)

        file_tup = self.get_files(idx - self._prev_doc_len)
        if file_tup:
            return self._label_chunk[idx - self._prev_doc_len], self.get_files(idx - self._prev_doc_len)
        else:
            return self._label_chunk[idx - self._prev_doc_len]
