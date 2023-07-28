import json
import os
import tarfile

import requests
from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil
import pathlib
from tqdm import tqdm
from BenchKit.Data.Datasets import ProcessorDataset
from BenchKit.Miscellaneous.User import create_dataset, get_post_url, \
    delete_dataset, get_current_dataset, get_chunk_count

megabyte = 1_024 ** 2
gigabyte = megabyte * 1024
terabyte = gigabyte * 1024
limit = 100 * megabyte
file_limit = 50_000


class UploadError(Exception):
    pass


def get_folder_size(dataset_path) -> int:
    size = 0

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            if name.endswith(".tar.gz"):
                size += os.path.getsize(os.path.join(root, name))

    return size


def get_dataset(chunk_class,
                cloud: bool,
                dataset_name: str,
                batch_size: int,
                num_workers: int,
                *args,
                **kwargs):

    dl = DataLoader(dataset=chunk_class(dataset_name,
                                        cloud,
                                        *args,
                                        **kwargs),
                    num_workers=num_workers,
                    batch_size=batch_size,
                    worker_init_fn=chunk_class.worker_init_fn)

    return dl


def remove_all_temps():
    for i in os.listdir("."):
        if i.startswith("Temp"):
            shutil.rmtree(os.path.join(".", i))


def upload_file(url,
                file_path,
                save_path,
                fields):
    with open(file_path, 'rb') as f:
        files = {'file': (save_path, f)}
        http_response = requests.post(url,
                                      data=fields,
                                      files=files)

    if http_response.status_code != 204:
        raise RuntimeError(f"Failed to Upload {file_path}")


def create_dataset_zips(processed_dataset: ProcessorDataset,
                        dataset_name: str):
    if os.path.isdir(f"ProjectDatasets/{dataset_name}"):
        shutil.rmtree(f"ProjectDatasets/{dataset_name}")

    print(Fore.RED + "Started Data processing" + Style.RESET_ALL)

    count = save_file_and_label(processed_dataset, dataset_name)

    print(Fore.GREEN + "Data is processed" + Style.RESET_ALL)

    ds = get_current_dataset(dataset_name)

    if ds:
        delete_dataset(ds["id"])

    create_dataset(dataset_name,
                   count,
                   get_folder_size(f"./ProjectDatasets/{dataset_name}"))

    # ds = get_current_dataset(dataset_name)

    # patch_dataset_list(ds["id"], count)


def test_dataloading(dataset_name: str,
                     chunk_dataset,
                     *args,
                     **kwargs):
    num_workers = 2
    batch_size = 16

    ds = get_current_dataset(dataset_name)

    if not ds:
        raise RuntimeError("Dataset must be created")

    length = ds["sample_count"]

    if length == 0:
        raise RuntimeError("Data has not been processed")

    dl = get_dataset(chunk_dataset,
                     False,
                     ds["name"],
                     batch_size,
                     num_workers,
                     *args,
                     **kwargs)

    print(Fore.RED + "Running Data Loading test" + Style.RESET_ALL)
    for _ in tqdm(dl, colour="blue", total=int(np.ceil(length / batch_size)) + 1):
        pass

    # remove_all_temps()

    print(Fore.GREEN + "Data Loading Test Passed" + Style.RESET_ALL)


def run_upload(dataset_name: str):
    ds = get_current_dataset(dataset_name)

    if not ds:
        raise RuntimeError("Dataset must be created")

    length = ds["sample_count"]

    if os.path.isdir(f"ProjectDatasets/{dataset_name}"):
        x = os.listdir(f"ProjectDatasets/{dataset_name}")
        if len(x) == 0:
            raise RuntimeError("Project Folder is empty")
    else:
        raise RuntimeError("Project Folder does not exist")

    if length == 0:
        raise RuntimeError("Data has not been processed")

    print(Fore.RED + "Started Upload" + Style.RESET_ALL)

    save_path = f"ProjectDatasets/{dataset_name}"

    last_file_number = get_chunk_count(ds["id"])

    for path in tqdm(iterate_directory(save_path, last_file_number),
                     total=(len(os.listdir(save_path)) - last_file_number),
                     colour="blue"):

        file_count: str = os.path.split(path)[-1]

        file_count: int = int(file_count.split("-")[-1][:-len(".tar.gz")])

        response = get_post_url(ds["id"],
                                os.path.getsize(path),
                                os.path.split(path)[-1],
                                file_count)

        resp = json.loads(response.content)
        upload_file(resp["url"],
                    path,
                    os.path.split(path)[-1],
                    resp["fields"])

        ds = get_current_dataset(dataset_name)

    print(Fore.GREEN + "Finished Upload" + Style.RESET_ALL)

    shutil.rmtree(f"ProjectDatasets/{dataset_name}")


def add_tar_directory(folder_path: str, tar_file: tarfile.TarFile):
    root_dir = os.path.basename(os.path.normpath(folder_path))
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            arc_name = os.path.relpath(file_path, folder_path)
            tar_file.add(file_path, arcname=os.path.join(root_dir, arc_name))


def get_tar_size(tar_file_path: str):
    return os.path.getsize(tar_file_path)


def save_folder_data(save_folder: str,
                     chunk_num: int,
                     label_batch: list,
                     input_batch: list,
                     proc_dat: ProcessorDataset,
                     count: int,
                     tar: tarfile.TarFile) -> int:
    folder_str = "dataset-{}"

    folder_path = os.path.join(save_folder, folder_str.format(chunk_num))
    os.makedirs(folder_path,
                exist_ok=True)

    dir_path = os.path.join(folder_path, f"ds-{count}")

    os.makedirs(dir_path, exist_ok=True)

    input_list = []
    label_list = []

    flag_list = [True] * len(input_batch)
    input_batch.extend(label_batch)
    flag_list += [False] * len(label_batch)

    for val, flag in zip(input_batch, flag_list):

        if len(val) == 2:

            p = proc_dat.validate(val[0],
                                  dir_path,
                                  dtype=val[1])
        else:
            p = proc_dat.validate(val[0],
                                  dir_path)

        if flag:
            input_list.append(p)
        else:
            label_list.append(p)

    ret_dict = {
        "input_list": [os.path.split(i)[-1] for i in input_list],
        "label_list": [os.path.split(i)[-1] for i in label_list],
    }

    with open(f"{dir_path}/ann.json", 'w') as f:
        json.dump(ret_dict, f)

    add_tar_directory(dir_path, tar)

    shutil.rmtree(dir_path)

    return len(input_list) + len(label_list)


def save_file_and_label(dataset: ProcessorDataset,
                        ds_name: str):
    cwd = os.getcwd()
    save_folder = os.path.join(cwd, "ProjectDatasets", ds_name)

    if os.path.isdir(save_folder):
        raise UploadError("Folder already exists")
    else:
        os.makedirs(save_folder)

    current_file_size = 0
    file_count = 0
    count = 0
    temp_count = 0
    chunk_num = 0

    tar_path = os.path.join(save_folder, f"dataset-{chunk_num}.tar.gz")
    tar = tarfile.open(tar_path, "w:gz", compresslevel=1, format=tarfile.PAX_FORMAT)

    for batch in tqdm(dataset,
                      colour="blue"):
        count += 1
        temp_count += 1
        input_list, label_list = batch

        if current_file_size <= limit and file_count <= file_limit:

            f_count = save_folder_data(save_folder,
                                       chunk_num,
                                       list(label_list),
                                       list(input_list),
                                       dataset,
                                       temp_count - 1,
                                       tar)

            current_file_size = get_tar_size(tar_path)

            file_count += f_count

        else:

            tar.close()
            os.rename(tar_path,
                      os.path.join(save_folder, f"dataset-{chunk_num}-{temp_count - 1}.tar.gz"))

            shutil.rmtree(os.path.join(save_folder, f"dataset-{chunk_num}"))

            chunk_num += 1
            tar_path = os.path.join(save_folder, f"dataset-{chunk_num}.tar.gz")
            tar = tarfile.open(tar_path, "w:gz", compresslevel=9, format=tarfile.PAX_FORMAT)

            f_count = save_folder_data(save_folder,
                                       chunk_num,
                                       list(label_list),
                                       list(input_list),
                                       dataset,
                                       0,
                                       tar)

            temp_count = 1
            file_count = f_count

            current_file_size = get_tar_size(tar_path)

    tar.close()
    os.rename(tar_path,
              os.path.join(save_folder, f"dataset-{chunk_num}-{temp_count}.tar.gz"))
    shutil.rmtree(os.path.join(save_folder, f"dataset-{chunk_num}"))

    return count


def get_dir_size(path) -> int:
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def merge_folders(small_folder: str, large_folder: str, save_folder: str):
    small_path = os.path.join(save_folder, os.path.split(small_folder)[-1].split(".")[0])
    large_path = os.path.join(save_folder, os.path.split(large_folder)[-1].split(".")[0])

    shutil.unpack_archive(small_folder, small_path)
    shutil.unpack_archive(large_folder, large_path)

    small_tensor = []
    large_tensor = []

    small_files = []
    large_files = []

    lt_path = ''
    lf_path = ''

    for i in os.listdir(small_path):
        tail = small_path.split(".")[0]
        pth = os.path.join(tail, i)
        if i.endswith(".pt"):
            small_tensor: list = torch.load(pth)
        else:
            small_files = [os.path.join(pth, i) for i in os.listdir(pth)]

    for i in os.listdir(large_path):
        tail = large_path.split(".")[0]
        pth = os.path.join(tail, i)
        if i.endswith(".pt"):
            lt_path = pth
            large_tensor: list = torch.load(pth)
        else:
            lf_path = pth
            large_files = [os.path.join(pth, i) for i in os.listdir(pth)]

    large_tensor.extend(small_tensor)

    if large_files:
        large_files = sorted(large_files, key=lambda x: int(x.split("-")[-1]))
        small_files = sorted(small_files, key=lambda x: int(x.split("-")[-1]))

        last_int = int(large_files[-1].split("-")[-1])

        with ThreadPoolExecutor(15) as exe:
            _ = [exe.submit(shutil.copytree,
                            i,
                            os.path.join(lf_path, f"file-{idx + last_int}")) for idx, i in
                 enumerate(small_files)]

    torch.save(large_tensor, lt_path)
    os.remove(small_folder)
    os.remove(large_folder)
    shutil.rmtree(small_path)

    head, tail = os.path.split(large_folder)

    f_name = tail.split(".")[0]

    f_list = f_name.split("-")
    f_name = f"dataset-{f_list[1]}-{len(large_tensor)}-zip"
    shutil.make_archive(f"{head}/{f_name}",
                        "zip",
                        large_path)

    shutil.rmtree(large_path)

    return large_folder


def affirm_size(save_folder: str):
    pass_size_requirement = []
    fails_size_requirement = []

    if get_dir_size(save_folder) < megabyte * 100:
        raise RuntimeError("Dataset must be greater than 100 megabytes")

    for i in os.listdir(save_folder):
        path: str = os.path.join(save_folder, i)

        if os.path.getsize(path) >= limit:
            pass_size_requirement += [path]
        else:
            fails_size_requirement += [path]

    while len(fails_size_requirement) != 1:
        large_folder = fails_size_requirement[0]
        small_folder = fails_size_requirement.pop()

        large_folder = merge_folders(small_folder, large_folder, save_folder)

        size = os.path.getsize(large_folder)

        if size >= limit:
            pass_size_requirement.append(fails_size_requirement.pop(0))

    if len(pass_size_requirement) > 0:

        while len(fails_size_requirement) > 0:
            small_folder = fails_size_requirement.pop()
            large_folder = pass_size_requirement.pop()

            large_folder = merge_folders(small_folder, large_folder, save_folder)

            pass_size_requirement.insert(0, large_folder)


def iterate_directory(file_dir: str,
                      current_file: int) -> tuple[str, bool]:
    for idx, i in enumerate(os.listdir(file_dir)):
        if idx >= current_file:
            yield str(pathlib.Path(file_dir).resolve() / i)


def create_dataset_dir():
    if os.path.isdir("./Datasets"):
        pass
    else:
        current_path = "./Datasets"
        os.mkdir(current_path)

        whole_path = os.path.join(current_path, "ProjectDatasets.py")
        init_path = os.path.join(current_path, "__init__.py")

        with open(init_path, "w"):
            pass

        with open(whole_path, "w") as file:
            file.write("from BenchKit.Data.Datasets import ProcessorDataset, IterableChunk\n")
            file.write("# Write your datasets or datapipes here")
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("def main():\n")
            file.write('    """\n')
            file.write("    This method returns all the necessary components to build your dataset\n")
            file.write("    You will return a list of tuples, each tuple represents a different dataset\n")
            file.write("    The elements of the tuple represent the components to construct your dataset\n")
            file.write("    Element one will be your ProcessorDataset\n")
            file.write("    Element two will be your IterableChunk\n")
            file.write("    Element three will be the name of your Dataset\n")
            file.write("    Element four will be all the args needed for your Iterable Chunk as a list\n")
            file.write("    Element five will be all the kwargs needed for your Iterable Chunk as a Dict\n")
            file.write('    """\n')
            file.write("    pass\n")
