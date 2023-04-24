import docker
import subprocess
from BenchKit.Data.Helpers import upload_file
from BenchKit.Miscellaneous.Settings import get_config, set_config
from BenchKit.Miscellaneous.BenchKit import write_config
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

    set_config({
        "docker_details": {
            "tarball_name": file_name + ".tar.gz",
            "image_name": docker_image_name
        }
    })

    os.remove("Dockerfile")
    os.remove("entrypoint.sh")

    write_config()


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


def save_image_tarball():
    docker_config = get_config()["docker_details"]

    image_name = docker_config["image_name"]

    docker_path = docker_config['tarball_name']

    subprocess.run(f"docker save {image_name} | gzip > {docker_path}", shell=True)

    return docker_path


def upload_tarball(version: int | None = None):
    if not version:
        version = 1

    docker_config = get_config()["docker_details"]
    tar_ball_name = docker_config["tarball_name"]

    size = os.path.getsize(tar_ball_name)

    url_data = project_image_upload_url(size,
                                        version,
                                        tar_ball_name)

    upload_file(url_data["url"],
                tar_ball_name,
                tar_ball_name,
                url_data["fields"])

    os.remove(tar_ball_name)
