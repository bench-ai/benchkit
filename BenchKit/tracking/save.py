import os
from .config import Config
from BenchKit.Miscellaneous.requests.model_save import post_model_state_presigned_url, post_model_save_presigned_url
from BenchKit.Miscellaneous.utils.tar import tar_gzip_folder
from BenchKit.Miscellaneous.utils.bucket import upload_using_presigned_url


def upload_model_save(save_folder_path: str,
                      tar_save_location: str,
                      evaluation_value: float,
                      config: Config) -> None:
    """
    Uploads model_save to Bench AI central repository

    :param save_folder_path: Path to the folder where the state is saved
    :type state_folder_path: str
    :param tar_save_location: Path where the tar file should be saved
    :type tar_save_location: str
    :param evaluation_value: float value in the respect to the config evaluation criteria
    :type evaluation_value: float
    :param config:
    :type config: Config
    """

    file_name = f"bench-{config.config_id}-save-{config.evaluation_criteria}.tar.gz"

    tar_save_path = tar_gzip_folder(file_name,
                                    save_folder_path,
                                    tar_save_location)

    tar_size_bytes = os.path.getsize(tar_save_path)

    presigned_post_url = post_model_save_presigned_url(config,
                                                       tar_size_bytes,
                                                       evaluation_value,)

    upload_using_presigned_url(presigned_post_url["url"],
                               tar_save_path,
                               os.path.basename(tar_save_path),
                               fields=presigned_post_url["fields"])


def upload_model_state(state_folder_path: str,
                       tar_save_location: str,
                       iteration: int,
                       evaluation_value: float,
                       config: Config) -> None:
    """
    Uploads model_state to Bench AI central repository

    :param state_folder_path: Path to the folder where the state is saved
    :type state_folder_path: str
    :param tar_save_location: Path where the tar file should be saved
    :type tar_save_location: str
    :param iteration: current state iteration
    :type iteration: int
    :param evaluation_value: float value in the respect to the config evaluation criteria
    :type evaluation_value: float
    :param config:
    :type config: Config
    """

    file_name = f"bench-{config.config_id}-{iteration}-{config.evaluation_criteria}.tar.gz"

    tar_save_path = tar_gzip_folder(file_name,
                                    state_folder_path,
                                    tar_save_location)

    tar_size_bytes = os.path.getsize(tar_save_path)

    presigned_post_url = post_model_state_presigned_url(config,
                                                        iteration,
                                                        evaluation_value,
                                                        tar_size_bytes)

    upload_using_presigned_url(presigned_post_url["url"],
                               tar_save_path,
                               os.path.basename(tar_save_path),
                               fields=presigned_post_url["fields"])
