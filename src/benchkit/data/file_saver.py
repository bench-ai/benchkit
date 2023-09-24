import json
import os
import pickle  # noqa S403
import shutil
import uuid
import warnings

import numpy as np
import torch


class BaseFile:
    def __init__(self):
        self._prefix = ""
        self._file_name = f"bench-{str(uuid.uuid4())}"

    def reset(self):
        self._file_name = f"bench-{str(uuid.uuid4())}"

    @property
    def file_name(self):
        return self._file_name

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError("load has not been implemented")

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str):
        self._prefix = prefix

    @property
    def save_path(self):
        return os.path.join(self.prefix, self._file_name)

    def save(self, *args, **kwargs) -> tuple[str, str]:
        raise NotImplementedError("Save is required to write the file to disk")

    def append(self, *args, **kwargs) -> None:
        raise NotImplementedError("Append is required to add to the data")

    def __call__(self, idx, *args, **kwargs):
        raise NotImplementedError("Append is required to add to the data")


class TextFile(BaseFile):
    def __init__(self, line_list: list | None = None):
        super().__init__()
        self._file_name = super().file_name + "-text.txt"
        self._line_list = line_list if line_list else []

    def save(self) -> tuple[str, str]:
        with open(self.save_path, "w") as file:
            file.writelines(self._line_list)

        return self.file_name, "textfile"

    def reset(self):
        super().reset()
        self._file_name = super().file_name + "-text.txt"
        self._line_list = []

    def append(self, line: str):
        if not line.endswith("\n"):
            line += "\n"

        self._line_list.append(line)

    @classmethod
    def load(cls, save_path: str):
        with open(save_path) as file:
            lines = file.readlines()

        return cls(line_list=lines)

    def __call__(self, idx, *args, **kwargs) -> str:
        return self._line_list[idx][:-1]


class BooleanFile(TextFile):
    def __init__(self, line_list: list | None = None):
        super().__init__(line_list=line_list)

    def append(self, line: bool):
        super().append(str(int(line)))

    def __call__(self, idx, *args, **kwargs) -> bool:
        return bool(int(super().__call__(idx)))

    def save(self):
        name, _ = super().save()
        return name, "bool"


class NumpyFile(BaseFile):
    def __init__(self, enforce_shape: bool, shape: tuple[int, ...] | None = None):
        super().__init__()
        self._enforce_shape = enforce_shape

        file_str = super().file_name
        file_str += "-{}"

        self._file_name = (
            file_str.format("enforced-array.npy")
            if enforce_shape
            else file_str.format("array.npz")
        )
        self.shape = shape

        self._arr_dict = {}
        self.arr = np.array([np.NaN])

    def reset(self):
        super().reset()
        file_str = super().file_name
        file_str += "-{}"
        self._file_name = (
            file_str.format("enforced-array.npy")
            if self.enforce_shape
            else file_str.format("array.npz")
        )
        self.arr_dict = {}
        self.arr = np.array([np.NaN])

    @property
    def arr_dict(self):
        return self._arr_dict

    @arr_dict.setter
    def arr_dict(self, arr_dict):
        self._arr_dict = arr_dict

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, arr):
        self._arr = arr

    @property
    def enforce_shape(self):
        return self._enforce_shape

    @enforce_shape.setter
    def enforce_shape(self, new_shape: tuple[int]):
        self._enforce_shape = new_shape

    def append(self, arr):
        if isinstance(arr, list):
            arr = np.array(arr)

        if self.enforce_shape:
            if arr.shape != self.shape:
                raise RuntimeError(
                    f"Enforced Shape of {self.shape} does not match array shape {arr.shape}"
                )

            arr = np.expand_dims(arr, axis=0)

            if np.isnan(self.arr).any():
                self.arr = arr
            else:
                self.arr = np.concatenate([self.arr, arr], axis=0)
        else:
            self.arr_dict[f"np-{len(self.arr_dict)}"] = arr

    def __call__(self, idx, *args, **kwargs):
        if self.enforce_shape:
            return self.arr[idx]
        else:
            return self.arr_dict[f"np-{idx}"]

    @classmethod
    def load(cls, save_path: str, enforce_shape: bool):
        instance = cls(enforce_shape)

        loaded_array = np.load(save_path, allow_pickle=True)

        if enforce_shape:
            instance.arr = loaded_array
        else:
            instance.arr_dict = loaded_array

        return instance

    def save(self):
        if self.enforce_shape:
            np.save(self.save_path, self.arr)

            return self.file_name, "enforced_arr"
        else:
            np.savez(self.save_path, **self.arr_dict)
            return self.file_name, "arr"


class TorchFile(BaseFile):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self._file_name = super().file_name + "-ten.pt"
        self.shape = torch.Size(shape)

        self._ten = None
        self._cat = False

    @property
    def ten(self):
        return self._ten

    @ten.setter
    def ten(self, ten: torch.Tensor):
        self._ten = ten

    def append(self, ten: torch.Tensor):
        if isinstance(ten, (list, np.ndarray)):
            ten = torch.Tensor(ten)
        elif isinstance(ten, torch.Tensor):
            pass
        else:
            raise ValueError(f"TorchFile does not accept {ten.__class__}")

        if ten.size() != self.shape:
            raise RuntimeError(
                f"Enforced Shape of {self.shape} does not match tensor shape {ten.size()}"
            )

        ten = torch.unsqueeze(ten, dim=0)

        if not self._cat:
            self.ten = ten
            self._cat = True
        else:
            self.ten = torch.cat((self.ten, ten), dim=0)

    def save(self):
        torch.save(self.ten, self.save_path)

        return self.file_name, "ten"

    @classmethod
    def load(cls, save_path: str):
        instance = cls((1, 1, 1))

        loaded_tensor = torch.load(save_path)

        instance.ten = loaded_tensor

        return instance

    def reset(self):
        super().reset()
        self._file_name = super().file_name + "-ten.pt"
        self.ten = None
        self._cat = False

    def __call__(self, idx, *args, **kwargs):
        return self.ten[idx]


class JsonFile(BaseFile):
    def __init__(self):
        super().__init__()
        self._json_list = []
        self._file_name = super().file_name + "-json.json"

    def append(self, app):
        self.json_list.append(app)

    def reset(self):
        super().reset()
        self.json_list = []
        self._file_name = super().file_name + "-json.json"

    @property
    def json_list(self):
        return self._json_list

    @json_list.setter
    def json_list(self, j_list: list):
        self._json_list = j_list

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.json_list, f)

        return self.file_name, "json"

    @classmethod
    def load(cls, save_path: str):
        with open(save_path) as file:
            j_list = json.load(file)

        instance = cls()

        instance.json_list = j_list

        return instance

    def __call__(self, idx, *args, **kwargs):
        return self.json_list[idx]


class NumericFile(BaseFile):
    def __init__(self):
        super().__init__()
        self._file_name = super().file_name + "-num.pkl"
        self._numeric_list = []

    @property
    def numeric_list(self):
        return self._numeric_list

    @numeric_list.setter
    def numeric_list(self, num_list: list[float | int]):
        self._numeric_list = num_list

    def reset(self):
        super().reset()
        self._file_name = super().file_name + "-num.pkl"
        self.numeric_list = []

    def append(self, number):
        self.numeric_list.append(number)

    def save(self):
        with open(self.save_path, "wb") as file:
            pickle.dump(self.numeric_list, file)

        return self.file_name, "num"

    @classmethod
    def load(cls, save_path: str):
        # TODO: Possible security issue as stated by S301.
        with open(save_path, "rb") as file:
            my_var = pickle.load(file)  # noqa S301

        instance = cls()
        instance.numeric_list = my_var

        return instance

    def __call__(self, idx, *args, **kwargs):
        return self.numeric_list[idx]


class RawFile(BaseFile):
    def __init__(self):
        super().__init__()
        self._file_list = []

    @property
    def file_list(self):
        return self._file_list

    @file_list.setter
    def file_list(self, file_list: list[str]):
        self._file_list = file_list

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, prefix):
        os.mkdir(os.path.join(prefix, self.file_name))
        self._prefix = prefix

    def reset(self):
        super().reset()
        self._file_list = []

    def append(self, file_path: str):
        file_name = os.path.split(file_path)[-1]
        self._file_list.append(file_name)

        if file_name in self._file_list:
            warnings.warn(f"Duplicate file named: {file_name} found", stacklevel=2)

        shutil.copyfile(file_path, os.path.join(self.save_path, file_name))

    def save(self):
        with open(os.path.join(self.save_path, "ann.json"), "w") as file:
            json.dump(self.file_list, file)

        return self.file_name, "folder"

    @classmethod
    def load(cls, save_path: str):
        with open(os.path.join(save_path, "ann.json")) as file:
            file_list = json.load(file)

        file_list = [os.path.join(save_path, i) for i in file_list]

        instance = cls()
        instance.file_list = file_list

        return instance

    def __call__(self, idx, *args, **kwargs):
        return self.file_list[idx]
