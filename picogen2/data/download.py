import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf
from termcolor import colored

from .. import utils
from ..utils import logger, song_dir_name


def pop2piano(song_file: Path, data_dir: Path, partition: Tuple[int, int] = (0, 1)):
    assert data_dir.exists(), f"{data_dir} does not exist"

    song_id_pairs = []
    for line in song_file.read_text().strip().split("\n")[1:]:
        song_id_pairs.append(line.split(","))
    song_id_pairs.sort()

    for i, (piano_id, song_id) in enumerate(song_id_pairs):
        if i % partition[1] != partition[0]:
            continue

        piano_url = f"https://www.youtube.com/watch?v={piano_id}"
        song_url = f"https://www.youtube.com/watch?v={song_id}"

        tgt_piano_file = data_dir / song_dir_name(i) / "piano.mp3"
        tgt_info_file = data_dir / song_dir_name(i) / "info.yaml"
        tgt_song_file = data_dir / song_dir_name(i) / "song.mp3"

        if tgt_piano_file.exists() and tgt_info_file.exists() and tgt_song_file.exists():
            logger.info(f"Skipping ({piano_id}, {song_id})")
            continue

        tmp_piano_file = tempfile.NamedTemporaryFile(suffix=".mp3")
        tmp_song_file = tempfile.NamedTemporaryFile(suffix=".mp3")

        try:
            logger.info(f"Downloading piano: {piano_url}")
            piano_meta = donwload_song(piano_url, Path(tmp_piano_file.name))

            logger.info(f"Downloading song: {song_url}")
            song_meta = donwload_song(song_url, Path(tmp_song_file.name))

            meta = OmegaConf.create()
            meta.piano = piano_meta
            meta.song = song_meta

            tgt_piano_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp_piano_file.name, tgt_piano_file)
            shutil.copy(tmp_song_file.name, tgt_song_file)
            OmegaConf.save(meta, tgt_info_file)

            logger.info(colored(f"Done ({piano_id}, {song_id})", "green"))
        except subprocess.CalledProcessError:
            logger.warn(colored(f"Failed to download ({piano_id}, {song_id})", "yellow"))


def donwload_song(url: str, output_file: Path):
    # get video metadata
    r = subprocess.run(
        [
            "yt-dlp",
            "--print-json",
            "--skip-download",
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    video_info = json.loads(r.stdout)
    meta = OmegaConf.create()
    meta.uploader = video_info["uploader"]
    meta.title = video_info["title"]
    meta.ytid = video_info["id"]
    meta.duration = int(video_info["duration"])
    meta.url = url

    # download video
    ext = "mp3"
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = Path(tmp_dir.name) / f"audio.{ext}"
    subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-quality",
            "0",
            "--retries",
            "10",
            "--audio-format",
            ext,
            "--postprocessor-args",
            "ffmpeg:-ac 2 -ar 44100",
            "-o",
            tmp_file,
            url,
        ],
        check=True,
    )
    output_file.write_bytes(tmp_file.read_bytes())

    return meta
