import docker
import subprocess
from BenchKit.Data.Helpers import upload_file
import os
from pathlib import Path
from BenchKit.Miscellaneous.User import project_image_upload_url, get_user_project

client = docker.from_env()


def build_docker_image(version: int | None = None):
    project = get_user_project()
    docker_path = Path(__file__).resolve().parent / "Dockerfile"

    if not os.path.isfile("requirements.txt"):
        raise RuntimeError("requirements.txt file is needed")

    with open(docker_path, "r") as f:
        with open("Dockerfile", "w") as file:
            line = f.readline()
            while line:
                file.write(line)
                line = f.readline()

    version = 1 if not version else version
    tag = "latest"

    project_name: str = project["id"].replace("-", "")

    file_name = f"Bench-{project_name}-V{version}-{tag}"

    name_list = file_name.split("-")
    docker_image_name = ""
    for i in name_list[:-1]:
        docker_image_name += f"{i}-"

    docker_image_name = docker_image_name[:-1] + f":{name_list[-1]}"

    docker_image_name = docker_image_name.lower()

    write_entrypoint_shell()

    client.images.build(dockerfile="Dockerfile",
                        tag=docker_image_name,
                        path=".")

    os.remove("Dockerfile")
    os.remove("entrypoint.sh")

    return file_name + ".tar.gz", docker_image_name


def write_entrypoint_shell():

    entry_point = Path(__file__).resolve().parent / "entrypoint.txt"
    with open(entry_point, "r") as r_file:
        with open("entrypoint.sh", "w") as w_file:
            w_file.writelines(r_file.readlines())


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

    os.remove(tarball_name)
