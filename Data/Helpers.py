import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import DataLoader
import shutil
import boto3.s3.transfer as s3transfer
from tqdm import tqdm
from Data.Datasets import ProcessorDataset

megabyte = 1_024 ** 2
gigabyte = megabyte * 1024
terabyte = gigabyte * 1024
limit = 750 * megabyte


class UploadError(Exception):
    pass


def copy_file(folder_path: str,
              file_folder: str,
              chunk_num: int,
              idx: int,
              files: list):
    os.makedirs(os.path.join(folder_path, file_folder.format(chunk_num), f"files-{idx}"))
    for inner_idx, j in enumerate(files):
        shutil.copyfile(j, os.path.join(folder_path,
                                        file_folder.format(chunk_num),
                                        f"files-{idx}",
                                        os.path.split(j)[-1]))


def save_folder_data(save_folder: str,
                     chunk_num: int,
                     label_batch: list,
                     file_batch: list):
    file_str = "dataset-labels-{}.pt"
    folder_str = "dataset-chunk-{}"
    zip_str = f"dataset-{chunk_num}-zip"
    file_folder = "dataset-files-folder-{}"

    folder_path = os.path.join(save_folder, folder_str.format(chunk_num))
    os.makedirs(folder_path)
    torch.save(label_batch, os.path.join(folder_path, file_str.format(chunk_num)))

    os.makedirs(os.path.join(folder_path, file_folder.format(chunk_num)))

    with ThreadPoolExecutor(15) as exe:
        _ = [exe.submit(copy_file,
                        folder_path,
                        file_folder,
                        chunk_num,
                        idx,
                        i) for idx, i in enumerate(file_batch)]

    shutil.make_archive(os.path.join(save_folder, zip_str),
                        "zip",
                        folder_path)

    shutil.rmtree(folder_path)


def save_file_and_label(dataset: ProcessorDataset,
                        save_folder: str):
    save_folder = f"./ProjectDatasets/{save_folder}"

    if os.path.isdir(save_folder):
        raise UploadError("Folder already exists")
    else:
        os.makedirs(save_folder)

    dataloader = DataLoader(dataset=dataset,
                            shuffle=True,
                            num_workers=4,
                            batch_size=1)
    chunk_num = 0

    label_batch = []
    file_batch = []
    current_file_size = 0

    for batch in tqdm(dataloader):

        labels, file = batch

        if current_file_size <= limit:
            current_file_size += int(np.sum([os.path.getsize(i) for i in file[0]]))
            file_batch.append(file[0])
            label_batch.append(labels)

        else:
            save_folder_data(save_folder,
                             chunk_num,
                             label_batch,
                             file_batch)

            file_batch = []
            label_batch = []
            current_file_size = 0
            current_file_size += int(np.sum([os.path.getsize(i) for i in file[0]]))
            file_batch.append(file[0])
            label_batch.append(labels)
            chunk_num += 1

    save_folder_data(save_folder,
                     chunk_num,
                     label_batch,
                     file_batch)


def save_label_data(dataset: ProcessorDataset,
                    save_folder: str) -> None:
    save_folder = f"./ProjectDatasets/{save_folder}"

    if os.path.isdir(save_folder):
        raise UploadError("Folder already exists")
    else:
        os.makedirs(save_folder)

    dataloader = DataLoader(dataset=dataset,
                            shuffle=True,
                            num_workers=4,
                            batch_size=1)

    arr_len = None
    arr = []

    chunk_num = 0
    folder_name = "dataset-chunk-{}"
    file_str = "dataset-labels-{}.pt"
    zip_str = "dataset-{}-zip"

    folder_path = os.path.join(save_folder, folder_name.format(chunk_num))

    for batch in tqdm(dataloader):

        if os.path.isdir(folder_path):
            arr.append(batch)

            if len(arr) - 1 == arr_len:
                torch.save(arr, os.path.join(folder_path, file_str.format(chunk_num)))
                chunk_num += 1
                folder_path = os.path.join(save_folder, folder_name.format(chunk_num))
                arr = []
        else:
            os.mkdir(folder_path)
            file_path = os.path.join(folder_path, file_str.format(chunk_num))
            torch.save([batch], file_path)
            f_size = os.path.getsize(file_path)
            arr_len = np.ceil(limit / f_size)
            arr.append(batch)
            os.remove(file_path)

    torch.save(arr, os.path.join(folder_path, file_str.format(chunk_num)))

    shutil.make_archive(os.path.join(save_folder, zip_str),
                        "zip",
                        folder_path)

    shutil.rmtree(folder_path)


# test in the morning
def affirm_size(save_folder: str):
    pass_size_requirement = []
    fails_size_requirement = []

    for i in os.listdir(save_folder):
        path: str = os.path.join(save_folder, i)

        if os.path.getsize(path) >= limit:
            pass_size_requirement += [path]
        else:
            fails_size_requirement += [path]

    while len(fails_size_requirement) > 0:
        small_folder = fails_size_requirement.pop()
        large_folder = pass_size_requirement.pop()

        small_path = os.path.join(save_folder, os.path.split(small_folder)[-1].split(".")[-1])
        large_path = os.path.join(save_folder, os.path.split(large_folder)[-1].split(".")[-1])

        shutil.unpack_archive(small_folder, small_path)
        shutil.unpack_archive(large_folder, large_path)

        small_tensor = []
        large_tensor = []

        small_files = []
        large_files = []

        lt_path = ''
        lf_path = ''

        for i in os.listdir(small_path):
            pth = os.path.join(small_folder, i)
            if i.endswith(".pt"):
                small_tensor: list = torch.load(pth)
            else:
                small_files = [os.path.join(pth, i) for i in os.listdir(pth)]

        for i in os.listdir(large_path):
            pth = os.path.join(large_folder, i)
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
                _ = [exe.submit(shutil.copyfile,
                                i,
                                os.path.join(os.path.split(lf_path)[0],
                                             f"file-{idx + last_int}")) for idx, i in enumerate(small_files)]

        torch.save(large_tensor, lt_path)

        os.remove(small_folder)
        os.remove(large_folder)
        shutil.rmtree(small_path)

        shutil.make_archive(large_folder,
                            "zip",
                            large_path)

        shutil.rmtree(large_path)

        pass_size_requirement.insert(0, large_folder)


# def iterate_directory(file_dir: str) -> tuple[str, bool]:
#     with os.scandir(file_dir) as walk:
#         for i in walk:
#             if os.path.isfile(i):
#                 yield str(pathlib.Path(file_dir).resolve() / i.name), False
#             elif os.path.isdir(i):
#                 new_path = pathlib.Path(file_dir).resolve() / i.name
#                 yield str(new_path) + "/", True
#                 yield from iterate_directory(new_path)


def create_dataset_dir():
    if os.path.isdir("./DataFeeder"):
        raise IsADirectoryError("Datasets directory already exists")
    else:
        current_path = "./Datasets"
        os.mkdir(current_path)

        whole_path = os.path.join(current_path, "ProjectDatasets.py")

        with open(whole_path, "w") as file:
            file.write("# Write your datasets or datapipes here")


def upload_file(session,
                bucketname,
                s3dir,
                file_list,
                progress_func,
                workers=10):
    s3client = session.client('s3')
    transfer_config = s3transfer.TransferConfig(
        use_threads=True,
        max_concurrency=workers,
        multipart_threshold=100 * megabyte,
        multipart_chunksize=16 * megabyte,
    )

    for src in file_list:
        dst = os.path.join(s3dir, os.path.basename(src))
        x = s3client.upload_file(src,
                                 bucketname,
                                 dst,
                                 Config=transfer_config,
                                 Callback=progress_func)
