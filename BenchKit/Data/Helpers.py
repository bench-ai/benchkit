import json
import os
import tarfile

import requests
from colorama import Fore, Style
import numpy as np
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
file_limit = 10_000


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


def get_directory_size(dataset_path) -> int:
    total_size = 0

    for dir_path, dir_names, filenames in os.walk(dataset_path):
        for f in filenames:
            fp = os.path.join(dir_path, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def get_total_file_count(directory):
    total_files = 0
    idx = 0

    for idx, (root, dirs, files) in enumerate(os.walk(directory)):
        if idx != 0:
            total_files += len(files)

    return 0 if idx == 0 else total_files


def save_file_and_label(dataset: ProcessorDataset,
                        ds_name: str,
                        check=100):

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
    mult = None
    f_mult = None

    os.mkdir(os.path.join(save_folder, f"dataset-{chunk_num}"))
    dataset.prefix = os.path.join(save_folder, f"dataset-{chunk_num}")
    dataset.prepare()

    for _ in tqdm(dataset,
                  colour="blue"):

        count += 1
        temp_count += 1

        if current_file_size <= limit and file_count <= file_limit:

            if count == check:
                for _ in dataset.save_savers():
                    pass

                mult = get_directory_size(dataset.prefix)
                f_mult = get_total_file_count(dataset.prefix)

            if mult:
                if count % check == 0:
                    current_file_size += mult

            if f_mult:
                if count % check == 0:
                    file_count += f_mult

        else:

            with open(os.path.join(dataset.prefix, "ann.json"), "w") as f:
                ann_list = []
                for name, tag in dataset.save_savers():
                    ann_list.append((name, tag))
                json.dump(ann_list, f)

            compress_directory(dataset.prefix, os.path.join(save_folder, f"dataset-{chunk_num}-{temp_count}.tar.gz"))
            chunk_num += 1
            temp_count = 0
            os.mkdir(os.path.join(save_folder, f"dataset-{chunk_num}"))
            dataset.prefix = os.path.join(save_folder, f"dataset-{chunk_num}")
            dataset.reset_savers()
            dataset.prepare()
            current_file_size = 0
            file_count = 0

    with open(os.path.join(dataset.prefix, "ann.json"), "w") as f:
        ann_list = []
        for name, tag in dataset.save_savers():
            ann_list.append((name, tag))

        json.dump(ann_list, f)

    compress_directory(dataset.prefix, os.path.join(save_folder, f"dataset-{chunk_num}-{temp_count}.tar.gz"))

    return count


def compress_directory(directory_path, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(directory_path, arcname="")

    shutil.rmtree(directory_path)


def get_dir_size(path) -> int:
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)

    return total


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
