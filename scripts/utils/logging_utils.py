from pathlib import Path
from datetime import datetime


def get_report_file():

    date_str = datetime.now().strftime(
        "%Y%m%d"
    )

    output_dir = Path(
        "../outputs/evaluations/chat_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_dir / f"{date_str}.txt"


def append_report(content):

    report_file = get_report_file()

    with open(
        report_file,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(content)