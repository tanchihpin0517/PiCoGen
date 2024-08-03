import shutil
import subprocess
from pathlib import Path

from .utils import logger

CACHE_DIR = Path.home() / ".cache" / "picogen2"

URL_MODEL = "https://www.dropbox.com/scl/fi/6pjc9950zeex35wnrqn8c/model_ft_00070000?rlkey=ynt5oc6ju0lack9qoycuaaiel&st=e121yub0&dl=0"
URL_VOCAB = "https://raw.githubusercontent.com/tanchihpin0517/PiCoGen/v2/assets/vocab.json"
URL_CONFIG = "https://raw.githubusercontent.com/tanchihpin0517/PiCoGen/v2/assets/config.json"


def default_cache_dir_decorator(func):
    def wrapper(*args, **kwargs):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)

    return wrapper


@default_cache_dir_decorator
def checkpoint_file():
    default_ckpt_file = CACHE_DIR / "model_ft_00070000"
    if not default_ckpt_file.exists():
        logger.warning("Download default model from {}".format(URL_MODEL))
        logger.warning("Save to {}".format(default_ckpt_file))

        _download(URL_MODEL, default_ckpt_file)

    return default_ckpt_file


@default_cache_dir_decorator
def vocab_file():
    default_vocab_file = CACHE_DIR / "vocab.json"
    if not default_vocab_file.exists():
        logger.warning("Download default vocab from {}".format(URL_VOCAB))
        logger.warning("Save to {}".format(default_vocab_file))

        _download(URL_VOCAB, default_vocab_file)

    return default_vocab_file


@default_cache_dir_decorator
def config_file():
    default_config_file = CACHE_DIR / "config.json"
    if not default_config_file.exists():
        logger.warning("Download default config from {}".format(URL_CONFIG))
        logger.warning("Save to {}".format(default_config_file))

        _download(URL_CONFIG, default_config_file)

    return default_config_file


def _download(url, output_file_path, verbose=True):
    if verbose:
        logger.info(f"Downloading {url} to {output_file_path}")

    if shutil.which("wget") is None:
        logger.error("wget is not installed. Please install wget to download the model.")
        raise FileNotFoundError("`wget` is not installed")

    try:
        subprocess.run(["wget", url, "-O", str(output_file_path)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download file from {url}: {e}")
        if output_file_path.exists():
            output_file_path.unlink()
        raise e
