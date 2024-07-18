import json
from pathlib import Path
from typing import Tuple

from mirtoolkit import beat_transformer, bytedance_piano_transcription

from ..utils import check_task_done, logger, mark_task_done, song_dir_name

TASK_TRANS = "transcribe"
TASK_BEAT = "beat"


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
    }
    if task == "all":
        for t in task_func:
            task_func[t](data_dir, output_dir, partition, debug, overwrite)
    else:
        task_func[task](data_dir, output_dir, partition, debug, overwrite)


def transcribe(
    data_dir: Path, output_dir: Path, partition: Tuple[int, int], debug=False, overwrite=False
):
    song_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    song_dirs = song_dirs[partition[0] :: partition[1]]

    if debug:
        song_dirs = song_dirs[:2]

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
    song_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    song_dirs = song_dirs[partition[0] :: partition[1]]

    if debug:
        song_dirs = song_dirs[:2]

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
            detect_beat_file(input_file, output_file)

        mark_task_done(TASK_BEAT, output_dir / song_dir_name(index))


def detect_beat_file(input_file: Path, output_file: Path):
    beat, downbeat = beat_transformer.detect(input_file)
    output_file.write_text(
        json.dumps({"beat": beat.tolist(), "downbeat": downbeat.tolist()}, indent=4)
    )
