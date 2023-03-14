import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
debug = False

settings_path = Path(__file__).resolve().parent / "Config.json"


def get_config() -> dict:
    config = {}
    with open(settings_path, "r") as file:
        config.update(json.load(file))

    return config


def get_credentials() -> tuple[str, str]:
    return get_config()["user_credentials"]["username"], get_config()["user_credentials"]["password"]


def get_main_url() -> str:
    return get_config()["api_url"]


def set_config(new_config: dict):
    old_config = get_config()
    old_config.update(new_config)

    with open(settings_path, "w") as file:
        json.dump(old_config, file)
