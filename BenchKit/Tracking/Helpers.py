import os.path
import shutil
import tarfile
from accelerate import Accelerator
from BenchKit.Data.Helpers import upload_file
from accelerate.tracking import on_main_process
from BenchKit.Miscellaneous.User import post_checkpoint_url


def upload_model_checkpoint(acc: Accelerator,
                            checkpoint_name: str,
                            checkpoint_folder="checkpoint"):

    save_checkpoint(acc)
    upload_checkpoint(checkpoint_name,
                      checkpoint_folder)


def save_checkpoint(acc: Accelerator):
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    acc.save_state(output_dir="checkpoint")


@on_main_process
def upload_checkpoint(checkpoint_name: str,
                      checkpoint_folder="checkpoint"):

    if "." in checkpoint_name:
        raise ValueError("Enter the checkpoint name alone dont add the extension")

    with tarfile.open(checkpoint_name + ".tar.gz", "w:gz") as tar:
        tar.add(checkpoint_folder, arcname=os.path.basename(checkpoint_folder))

    req = post_checkpoint_url(checkpoint_name + ".tar.gz")

    upload_file(req["url"],
                checkpoint_name + ".tar.gz",
                checkpoint_name + ".tar.gz",
                req["fields"])

    shutil.rmtree(checkpoint_folder)
    os.remove(checkpoint_name + ".tar.gz")
