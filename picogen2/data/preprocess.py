import json
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numpy as np
from mirtoolkit import beat_transformer, bytedance_piano_transcription, sheetsage

from ..utils import check_task_done, logger, mark_task_done, song_dir_name
from .align import save_delayed_song

TASK_TRANS = "transcribe"
TASK_BEAT = "beat"
TASK_SHEETSAGE = "sheetsage"
TASK_ALIGN = "align"


def pop2piano(
    task,
    data_dir: Path,
    output_dir: Path,
    partition: Tuple[int, int] = (0, 1),
    debug=False,
    overwrite=False,
):
    assert data_dir.exists(), f"{data_dir} does not exist"
    assert output_dir.exists(), f"{output_dir} does not exist"

    logger.info(f"Data directory: {data_dir.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Partition: {partition}")

    task_func = {
        TASK_TRANS: transcribe,
        TASK_BEAT: detect_beat,
        TASK_SHEETSAGE: extract_sheetsage_last_hidden_state,
        TASK_ALIGN: align,
    }
    if task == "all":
        for t in task_func:
            task_func[t](data_dir, output_dir, partition, debug, overwrite)
    else:
        task_func[task](data_dir, output_dir, partition, debug, overwrite)


def _get_song_dirs(data_dir: Path, partition: Tuple[int, int], debug=False):
    song_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    song_dirs = song_dirs[partition[0] :: partition[1]]

    if debug:
        song_dirs = song_dirs[:2]

    return song_dirs


def transcribe(
    data_dir: Path, output_dir: Path, partition: Tuple[int, int], debug=False, overwrite=False
):
    song_dirs = _get_song_dirs(data_dir, partition, debug)

    for song_dir in song_dirs:
        index = int(song_dir.name)
        if not overwrite and check_task_done(TASK_TRANS, output_dir / song_dir_name(index)):
            logger.warn(f"Song {index} has been transcribed. Skip.")
            continue

        input_file = song_dir / "piano.mp3"
        output_file = output_dir / song_dir_name(index) / "piano.mid"

        logger.info(f"Transcribe song {index} from {input_file} to {output_file}")
        (output_dir / song_dir_name(index)).mkdir(parents=True, exist_ok=True)

        trancribe_file(input_file, output_file)
        mark_task_done(TASK_TRANS, output_dir / song_dir_name(index))


def trancribe_file(input_file: Path, output_file: Path):
    bytedance_piano_transcription.transcribe(input_file, output_file)


def detect_beat(
    data_dir: Path, output_dir: Path, partition: Tuple[int, int], debug=False, overwrite=False
):
    song_dirs = _get_song_dirs(data_dir, partition, debug)

    for song_dir in song_dirs:
        index = int(song_dir.name)
        if not overwrite and check_task_done(TASK_BEAT, output_dir / song_dir_name(index)):
            logger.warn(f"Song {index} has been beat detected. Skip.")
            continue

        logger.info("Detect beat for song %d", index)
        (output_dir / song_dir_name(index)).mkdir(parents=True, exist_ok=True)

        for name in ["piano", "song"]:
            input_file = song_dir / f"{name}.mp3"
            output_file = output_dir / song_dir_name(index) / f"{name}_beat.json"
            logger.info(f"Detecting beat for {input_file} to {output_file}")

            beats, downbeats = beat_transformer.detect(input_file)
            output_file.write_text(
                json.dumps({"beats": beats.tolist(), "downbeats": downbeats.tolist()}, indent=4)
            )

        mark_task_done(TASK_BEAT, output_dir / song_dir_name(index))


def extract_sheetsage_last_hidden_state(
    data_dir: Path, output_dir: Path, partition: Tuple[int, int], debug=False, overwrite=False
):
    song_dirs = _get_song_dirs(data_dir, partition, debug)
    for song_dir in song_dirs:
        index = int(song_dir.name)
        if not overwrite and check_task_done(TASK_SHEETSAGE, output_dir / song_dir_name(index)):
            logger.warn(f"Task {index} has been done. Skip.")
            continue

        logger.info("Extracting SheetSage's last hidden state for song %d", index)
        (output_dir / song_dir_name(index)).mkdir(parents=True, exist_ok=True)

        for name in ["piano", "song"]:
            input_file = song_dir / f"{name}.mp3"
            output_file = output_dir / song_dir_name(index) / f"{name}_sheetsage.npz"
            beat_file = output_dir / song_dir_name(index) / f"{name}_beat.json"

            beat_information = json.loads(beat_file.read_text())
            sheetsage_output = sheetsage.infer(
                audio_path=input_file, beat_information=beat_information
            )
            np.savez_compressed(
                output_file,
                melody=sheetsage_output["melody_last_hidden_state"],
                harmony=sheetsage_output["harmony_last_hidden_state"],
            )

        mark_task_done(TASK_SHEETSAGE, output_dir / song_dir_name(index))


def align(
    data_dir: Path,
    output_dir: Path,
    partition: Tuple[int, int],
    debug=False,
    overwrite=False,
):
    song_dirs = _get_song_dirs(data_dir, partition, debug)
    with mp.Pool() as pool:
        pool.starmap(_align_song, [(song_dir, output_dir, overwrite) for song_dir in song_dirs])


def _align_song(song_dir: Path, output_dir: Path, overwrite=False):
    index = int(song_dir.name)
    if not overwrite and check_task_done(TASK_ALIGN, output_dir / song_dir_name(index)):
        logger.warn(f"Task {index} has been done. Skip.")
        return

    logger.info("Align song %d", index)
    (output_dir / song_dir_name(index)).mkdir(parents=True, exist_ok=True)

    song_audio_file = song_dir / "song.mp3"
    piano_midi_file = output_dir / song_dir_name(index) / "piano.mid"
    align_info_file = output_dir / song_dir_name(index) / "align_info.json"

    save_delayed_song(
        song_audio_file=song_audio_file,
        piano_midi_file=piano_midi_file,
        align_info_file=align_info_file,
    )

    mark_task_done(TASK_ALIGN, output_dir / song_dir_name(index))
