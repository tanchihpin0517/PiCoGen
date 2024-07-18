import logging
from pathlib import Path

_logger = None
_level = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("picogen2")

    return _logger


class Logger:
    def setLevel(self, level):
        global _level, _logger
        _level = level.upper()
        _get_logger().setLevel(_level)

    def __getattr__(self, name):
        return getattr(_get_logger(), name)

    def __repr__(self):
        return repr(_get_logger())


logger = Logger()


def check_task_done(task: str, output_dir: Path):
    done_file = output_dir / f"done_{task}"
    return done_file.exists()


def mark_task_done(task: str, output_dir: Path):
    done_file = output_dir / f"done_{task}"
    done_file.touch()


def song_dir_name(index: int):
    return "{:04d}".format(index)
