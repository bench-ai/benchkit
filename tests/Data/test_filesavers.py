import os
import shutil
import uuid
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from BenchKit.Data.FileSaver import NumpyFile, BooleanFile, TextFile, TorchFile, JsonFile, NumericFile, RawFile, \
    BaseFile
from BenchKit.Data.Datasets import IterableChunk, ProcessorDataset
from BenchKit.Data.Helpers import save_file_and_label, get_test_dataloader
import random


class SingleClassTestProcessor(ProcessorDataset):

    def __init__(self):
        super().__init__()
        self.b1 = BooleanFile()

    def _get_savers(self) -> tuple[BaseFile, ...] | BaseFile:
        return self.b1

    def __len__(self):
        return 1_000

    def _get_data(self, idx: int):
        self.b1.append(True)


class SingleClassChunker(IterableChunk):

    def __init__(self):
        super().__init__()

    def unpack_data(self,
                    idx: int):
        bool = super().unpack_data(idx)
        return int(bool)


class MultiClassTestProcessor(ProcessorDataset):

    def __init__(self):
        super().__init__()
        self.nf1 = NumericFile()
        self.np1 = NumpyFile(enforce_shape=True, shape=(3, 100, 100))
        self.np2 = NumpyFile(enforce_shape=False)
        self.b1 = BooleanFile()
        self.t1 = TextFile()
        self.pt1 = TorchFile(shape=(4, 50, 20))
        self.j1 = JsonFile()
        self.r1 = RawFile()

    def _get_savers(self) -> tuple[BaseFile, ...] | BaseFile:
        return self.nf1, self.np1, self.np2, self.b1, self.t1, self.pt1, self.j1, self.r1

    def __len__(self):
        return 1_000

    def _get_data(self, idx: int):
        self.nf1.append(random.randint(1, 1000) if (idx % 2) == 0 else float(random.randint(1, 1000)))

        if idx % 2 == 0:
            arr = [0] * 100
            arr = [arr] * 100
            arr = [arr, arr, arr]
            self.np1.append(arr)
        else:
            self.np1.append(np.ones(shape=(3, 100, 100), dtype=np.float32))

        shape = (random.randint(1, 3), random.randint(1, 10), random.randint(1, 10))
        self.np2.append(np.ones(shape=shape, dtype=np.float32))
        self.b1.append((idx % 2) == 0)
        self.t1.append("Whats cookin good lookin")

        if idx % 2 == 0:
            self.pt1.append(torch.ones(size=(4, 50, 20)))
        elif idx % 2 == 1:
            self.pt1.append(np.ones(shape=(4, 50, 20)))
        else:
            arr = [0] * 20
            arr = [arr] * 50
            arr = [arr, arr, arr, arr]
            self.pt1.append(arr)

        self.j1.append([[{"test": "test"}] * 3])
        cred_path = Path(__file__).resolve().parent / "temp.txt"
        self.r1.append(str(cred_path))


class MultiClassChunker(IterableChunk):

    def __init__(self):
        super().__init__()

    def unpack_data(self,
                    idx: int):
        number, np1, np2, b1, t1, pt1, j1, r1 = super().unpack_data(idx)

        assert isinstance(number, (int, float))
        assert np1.shape == (3, 100, 100)
        assert np2.shape != (3, 100, 100)
        assert len(np2.shape) == 3
        assert isinstance(b1, bool)
        assert t1 == "Whats cookin good lookin"
        assert tuple(pt1.size()) == (4, 50, 20)
        assert isinstance(j1, list)
        assert os.path.isfile(r1)

        return number, np1, np2, b1, t1, pt1, j1, r1


def single_file_savers():
    ds_name = f"{str(uuid.uuid4())}-ds"
    processor = SingleClassTestProcessor()
    save_file_and_label(processor, ds_name=ds_name)

    scc = SingleClassChunker()

    dl = get_test_dataloader(scc,
                             ds_name,
                             ds_len=1_000)

    for _ in tqdm(dl):
        pass

    shutil.rmtree(os.path.join("ProjectDatasets", ds_name))


def all_file_savers():
    ds_name = f"{str(uuid.uuid4())}-ds"
    processor = MultiClassTestProcessor()
    save_file_and_label(processor, ds_name=ds_name)

    mcc = MultiClassChunker()
    dl = get_test_dataloader(mcc,
                             ds_name,
                             ds_len=1_000)

    for _ in tqdm(dl):
        pass

    shutil.rmtree(os.path.join("ProjectDatasets", ds_name))


def test_all_file_savers():
    all_file_savers()


def test_one_file_saver():
    single_file_savers()

## test custom file saver
