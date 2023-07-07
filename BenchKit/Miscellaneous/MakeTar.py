import tarfile
import os


def generate_tar(tar_name: str, file_path: str):
    with tarfile.open(f"{tar_name}.tar.gz", "w:gz") as tar:
        tar.add(file_path, arcname=os.path.split(file_path)[-1])

    return f"{tar_name}.tar.gz"


def extract_tar(tar_path: str, extraction_path: str):
    file = tarfile.open(tar_path)
    file.extractall(extraction_path)

