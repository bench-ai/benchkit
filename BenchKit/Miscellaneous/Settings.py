from datetime import datetime
from dateutil import tz
from dateutil import parser

def get_main_url() -> str:
    return "http://localhost:8000"
    # return "https://api.bench-ai.com"

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




