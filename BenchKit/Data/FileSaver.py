import json
import os
import pickle
import uuid
import numpy as np
import torch


class BaseFile:

    def __init__(self):
        self.prefix = ""
        self.file_name = ""
        raise NotImplementedError("Init Method is required to initialize the necessary parts of the file")

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError("load has not been implemented")

    @property
    def prefix(self):
        return self.prefix

    @prefix.setter
    def prefix(self, prefix: str):
        self.prefix = prefix

    @property
    def save_path(self):
        return os.path.join(self.prefix,
                            self.file_name)

    def save(self, *args, **kwargs) -> tuple[str, str]:
        raise NotImplementedError("Save is required to write the file to disk")

    def append(self, *args, **kwargs) -> None:
        raise NotImplementedError("Append is required to add to the data")

    def __call__(self, idx, *args, **kwargs):
        raise NotImplementedError("Append is required to add to the data")


class TextFile(BaseFile):

    def __init__(self,
                 line_list: list | None = None):

        super().__init__()
        self.file_name = f"bench-{str(uuid.uuid4())}-text.txt"
        self._file = open(self.save_path, "w")
        self._line_list = line_list

    def save(self) -> tuple[str, str]:
        self._file.close()
        return self.file_name, "textfile"

    def append(self, line: str):
        self._file.write(line)

    @classmethod
    def load(cls, save_path: str):
        with open(save_path, "r") as file:
            lines = file.readlines()

        return cls(line_list=lines)

    def __call__(self, idx, *args, **kwargs) -> str:
        return self._line_list[idx]


class BooleanFile(TextFile):

    def __init__(self,
                 line_list: list | None = None):

        super().__init__(line_list=line_list)

    def append(self, line: bool):
        super().append(str(int(line)))

    def __call__(self, idx, *args, **kwargs) -> bool:
        return bool(int(super()(idx)))


class NumpyFile(BaseFile):

    def __init__(self,
                 enforce_shape: bool,
                 shape: tuple[int] | None = None):

        super().__init__()
        self.enforce_shape = enforce_shape

        file_str = f"bench-{str(uuid.uuid4())}"
        file_str += "-{}"

        self.file_name = file_str.format("enforced-array.npy") if enforce_shape else file_str.format("array.npz")
        self.shape = shape

        if enforce_shape and not shape:
            raise RuntimeError("To enforce a shape a shape must be provided")

        self.arr_dict = {}
        self.arr = None

    @property
    def arr_dict(self):
        return self.arr_dict

    @arr_dict.setter
    def arr_dict(self, arr_dict):
        self.arr_dict = arr_dict

    @property
    def arr(self):
        return self.arr

    @arr.setter
    def arr(self, arr):
        self.arr = arr

    @property
    def enforce_shape(self):
        return self.enforce_shape

    @enforce_shape.setter
    def enforce_shape(self, new_shape: tuple[int]):
        self.enforce_shape = new_shape

    def append(self, arr):

        if isinstance(arr, list):
            arr = np.array(list)

        if self.enforce_shape:

            if arr.shape != self.shape:
                raise RuntimeError(f"Enforced Shape of {self.shape} does not match array shape {arr.shape}")

            arr = np.expand_dims(arr, axis=0)

            if not self.arr:
                self.arr = arr
            else:
                self.arr = np.concatenate([self.arr, arr], axis=0)
        else:
            self.arr_dict[f"np-{len(self.arr_dict)}"] = arr

    def __call__(self, idx, *args, **kwargs):

        if self.enforce_shape:
            return self.arr[idx]
        else:
            return self.arr_dict[f"np-{len(idx)}"]

    @classmethod
    def load(cls,
             save_path: str,
             enforce_shape: bool):

        instance = cls(enforce_shape)

        loaded_array = np.load(save_path, allow_pickle=True)

        if enforce_shape:
            instance.arr = loaded_array
        else:
            instance.arr_dict = loaded_array

        return instance

    def save(self):

        if self.enforce_shape:
            np.save(self.save_path,
                    self.arr)

            return self.save_path, "enforced_arr"
        else:
            np.savez(self.save_path, **self.arr_dict)
            return self.save_path, "arr"


class TorchFile(BaseFile):

    def __init__(self,
                 shape: tuple[int,...]):

        super().__init__()
        self.file_name = f"bench-{str(uuid.uuid4())}-ten.pt"
        self.shape = torch.Size(shape)

        self.ten = None

    @property
    def ten(self):
        return self.ten

    @ten.setter
    def ten(self, ten: torch.Tensor):
        self.ten = ten

    def append(self, ten: torch.Tensor):

        if isinstance(ten, list):
            ten = torch.Tensor(list)

        if ten.size() != self.shape:
            raise RuntimeError(f"Enforced Shape of {self.shape} does not match tensor shape {ten.size()}")

        ten = torch.unsqueeze(ten, dim=0)

        if not self.ten:
            self.ten = ten
        else:
            self.ten = torch.cat((self.ten, ten), dim=0)

    def save(self):
        torch.save(self.ten, self.save_path)

        return self.save_path, "ten"

    @classmethod
    def load(cls,
             save_path: str):

        instance = cls((1, 1, 1))

        loaded_tensor = torch.load(save_path)

        instance.ten = loaded_tensor

        return instance

    def __call__(self, idx, *args, **kwargs):
        return self.ten[idx]


class JsonFile(BaseFile):

    def __init__(self):

        super().__init__()
        self.json_list = []
        self.file_name = f"bench-{str(uuid.uuid4())}-json.json"

    def append(self, app):
        self.json_list.append(app)

    @property
    def json_list(self):
        return self.json_list

    @json_list.setter
    def json_list(self, j_list: list):
        self.json_list = j_list

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.json_list, f)

        return self.save_path, "json"

    @classmethod
    def load(cls,
             save_path: str):

        with open(save_path, "r") as file:
            j_list = json.load(file)

        instance = cls()

        instance.json_list = j_list

        return instance

    def __call__(self, idx, *args, **kwargs):
        return self.json_list[idx]


class NumericFile(BaseFile):

    def __init__(self):
        super().__init__()
        self.file_name = f"bench-{str(uuid.uuid4())}-json.json"
        self.numeric_list = []

    @property
    def numeric_list(self):
        return self.numeric_list

    @numeric_list.setter
    def numeric_list(self, num_list:list[float | int]):
        self.numeric_list = num_list

    def append(self, number):
        self.numeric_list.append(number)

    def save(self):
        with open(self.save_path, 'wb') as file:
            pickle.dump(self.numeric_list, file)

        return self.save_path, "num"

    @classmethod
    def load(cls,
             save_path: str):

        with open(save_path, 'rb') as file:
            my_var = pickle.load(file)

        instance = cls()
        instance.numeric_list = my_var

        return instance

    def __call__(self, idx, *args, **kwargs):
        return self.numeric_list[idx]


