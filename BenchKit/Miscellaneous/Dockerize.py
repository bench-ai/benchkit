import docker
import subprocess
from BenchKit.Data.Helpers import upload_file
from BenchKit.Miscellaneous.Settings import get_config, set_config
from BenchKit.Miscellaneous.BenchKit import write_config, update_code_version_config
import os
from pathlib import Path
from tqdm import tqdm
from BenchKit.Miscellaneous.User import get_gpu_count, project_image_upload_url

client = docker.from_env()


def build_docker_image(docker_image_path: str | None = None,
                       image_name: str | None = None,
                       version: int | None = None,
                       tag: str | None = None):
    cfg_project_name = get_config()["project"]["name"]
    docker_path = Path(__file__).resolve().parent / "Dockerfile"

    if not os.path.isfile("requirements.txt"):
        raise RuntimeError("requirements.txt file is needed")

    if not os.path.isfile("Dockerfile") and not docker_image_path:

        with open(docker_path, "r") as f:
            with open("Dockerfile", "w") as file:
                line = f.readline()
                while line:
                    file.write(line)
                    line = f.readline()

    version = 1 if not version else version
    tag = "latest" if not tag else tag

    default_name = f"Bench-{cfg_project_name}-V{version}-{tag}"
    file_name = image_name if image_name else default_name

    if not image_name:
        name_list = file_name.split("-")
        docker_image_name = ""
        for i in name_list[:-1]:
            docker_image_name += f"{i}-"

        docker_image_name = docker_image_name[:-1] + f":{name_list[-1]}"
    else:
        docker_image_name = f"{image_name}:{tag}"

    docker_image_name = docker_image_name.lower()

    write_entrypoint_shell()

    client.images.build(dockerfile="Dockerfile",
                        tag=docker_image_name,
                        path=".")

    os.remove("Dockerfile")
    os.remove("entrypoint.sh")

    return file_name + ".tar.gz", docker_image_name


def write_entrypoint_shell():
    gpu_dict = get_gpu_count()

    prefix_flags = ["--mixed_precision no"]

    if gpu_dict["multi"]:
        prefix_flags.append("--multi_gpu")

    prefix_flags.append("--num_machines 1")
    prefix_flags.append(f"--num_processes {gpu_dict['gpu_count']}")

    accelerate_string = 'accelerate '

    for i in prefix_flags:
        accelerate_string += f"{i} "

    accelerate_string += "launch TrainScript.py"

    with open("entrypoint.sh", "w") as file:

        file.write("#!/bin/sh" + "\n")
        file.write("pip install -r requirements.txt" + "\n")
        file.write("bench-kit -ss" + "\n")
        file.write(accelerate_string + "\n")


def save_image_tarball(tarball_name: str, image_name: str):
    subprocess.run(f"docker save {image_name} | gzip > {tarball_name}", shell=True)


def upload_tarball(tarball_name: str,
                   version: int | None = None):
    if not version:
        version = 1

    size = os.path.getsize(tarball_name)

    url_data = project_image_upload_url(size,
                                        version,
                                        tarball_name)

    upload_file(url_data["url"],
                tarball_name,
                tarball_name,
                url_data["fields"])

    update_code_version_config()
    os.remove(tarball_name)
