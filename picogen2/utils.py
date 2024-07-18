import logging

_logger = None
_level = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("picogen2")
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        if _level is not None:
            _logger.setLevel(_level)
    return _logger


class Logger:
    def setLevel(self, level):
        global _level, _logger
        _level = level.upper()
        if _logger is not None:
            _logger.setLevel(_level)

    def __getattr__(self, name):
        return getattr(_get_logger(), name)

    def __repr__(self):
        return repr(_get_logger())


logger = Logger()
