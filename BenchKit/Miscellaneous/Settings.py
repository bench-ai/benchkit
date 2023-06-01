import json
from dateutil import parser
from pathlib import Path

settings_path = Path(__file__).resolve().parent / "Config.json"


def get_config() -> dict:
    config = {}
    try:
        with open(settings_path, "r") as file:
            config.update(json.load(file))
    except FileNotFoundError:
        pass

    return config


# def get_credentials() -> tuple[str, str]:
#     return get_config()["user_credentials"]["username"], get_config()["user_credentials"]["password"]


def get_main_url() -> str:
    return "http://localhost:8000"
    # return "https://api.bench-ai.com"


def set_config(new_config: dict):
    old_config = get_config()
    old_config.update(new_config)

    with open(settings_path, "w") as file:
        json.dump(old_config, file)


def convert_iso_time(iso_time_str: str):
    return parser.parse(iso_time_str)
