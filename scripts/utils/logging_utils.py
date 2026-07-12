from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluations" / "chat_results"


def get_report_file(timestamp: datetime | None = None) -> Path:
    """Return the daily chat report file, independent of current working dir."""

    timestamp = timestamp or datetime.now()
    date_str = timestamp.strftime("%Y%m%d")

    CHAT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    return CHAT_RESULTS_DIR / f"{date_str}.txt"


def append_report(content: str, *, ensure_newline: bool = False) -> Path:
    """Append content to the daily report and return the file path.

    Returning the path makes callers easier to test or inspect, while existing
    callers can keep ignoring the result.
    """

    report_file = get_report_file()
    text = f"{content}\n" if ensure_newline and not content.endswith("\n") else content

    with report_file.open("a", encoding="utf-8") as file:
        file.write(text)

    return report_file
