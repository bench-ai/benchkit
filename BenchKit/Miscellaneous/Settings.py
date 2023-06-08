import json
from datetime import datetime
from dateutil import tz
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


def get_main_url() -> str:
    return "http://localhost:8000"
    # return "https://api.bench-ai.com"



def set_config(new_config: dict):
    old_config = get_config()
    old_config.update(new_config)

    with open(settings_path, "w") as file:
        json.dump(old_config, file, indent=4)


def convert_iso_time(iso_time_str: str):
    return parser.parse(iso_time_str)


def convert_timestamp(ts: str):

    dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%fZ')
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    dt = dt.replace(tzinfo=from_zone)
    dt_local = dt.astimezone(to_zone)

    local_time = dt_local.strftime('%Y-%m-%d %I:%M:%S %p')

    return local_time




