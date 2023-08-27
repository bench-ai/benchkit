import tarfile
import os


def tar_gzip_folder(tar_name: str,
                    folder_path: str,
                    tar_save_location: str = ".") -> str:
    """
    Tar's and GZIP's a directory and returns the path to it

    :param tar_name: the name you wish the tar file to have
    :param folder_path:
    :param tar_save_location: where you want the folder to be saved
    :return: the path to the gzip
    """

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a directory")

    tar_save_path: str = os.path.join(tar_save_location, tar_name)

    with tarfile.open(tar_save_path, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))

    return tar_save_path


def generate_tar(tar_name: str, file_path: str):
    with tarfile.open(f"{tar_name}.tar.gz", "w:gz") as tar:
        tar.add(file_path, arcname=os.path.split(file_path)[-1])

    return f"{tar_name}.tar.gz"


def extract_tar(tar_path: str, extraction_path: str):
    file = tarfile.open(tar_path)
    file.extractall(extraction_path)
