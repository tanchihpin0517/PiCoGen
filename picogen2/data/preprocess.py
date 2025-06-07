import json
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import numpy as np
from mirtoolkit.sheetsage import SheetSage
from mirtoolkit.bytedance_piano_transcription import ByteDancePianoTranscription
from mirtoolkit.beat_this import BeatThis
import pretty_midi

from ..utils import check_task_done, logger, mark_task_done, song_dir_name
from .align import save_delayed_song
from ..eval import melody_chroma_accuracy, _evaluate_melody, _f0

import librosa
import soundfile as sf
import torch
from pydub import AudioSegment
import onnxruntime as ort
import os
from .separate_fast import Predictor

# Добавляем импорт audio-separator
from audio_separator.separator import Separator

import matplotlib.pyplot as plt

TASK_TRANS = "transcribe"
TASK_BEAT = "beat"
TASK_SHEETSAGE = "sheetsage"
TASK_ALIGN = "align"
TASK_VOCAL = "vocal"
TASK_MEL_CHROMA = "mca"
TASK_MEL_CHROMA_VISUALIZE = "mca_visualize"

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
        TASK_VOCAL: extract_vocal,
        TASK_MEL_CHROMA: compute_melody_chroma_accuracy,
        TASK_MEL_CHROMA_VISUALIZE: visualize_melody_chroma,
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
    bytedance_piano_transcription = ByteDancePianoTranscription()
    bytedance_piano_transcription(input_file, output_file)

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

            beat_this = BeatThis()
            beats, downbeats = beat_this(input_file)
            if beats is None or downbeats is None:
                logger.warn(f"Beat detection failed or file missing for {input_file}, skipping.")
                continue
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

            if not beat_file.exists():
                logger.warn(f"Beat file {beat_file} does not exist, skipping {input_file}.")
                continue

            beat_information = json.loads(beat_file.read_text())
            sheetsage = SheetSage()
            sheetsage_output = sheetsage(
                audio_path=input_file, beat_information=beat_information
            )
            if sheetsage_output is None:
                logger.warn(f"SheetSage output is empty for {input_file}, skipping.")
                continue
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

def extract_vocal(
    data_dir: Path,
    output_dir: Path,
    partition: Tuple[int, int],
    debug: bool = False,
    overwrite: bool = False,
):
    overwrite = True
    """
    Итерируется по папкам датасета, ищет song.mp3, выделяет вокал с помощью audio-separator 
    и сохраняет результат как vocals.wav в выходной папке.
    """
    song_dirs = _get_song_dirs(data_dir, partition, debug)
    
    # Инициализируем audio-separator один раз для всех файлов
    separator = Separator(
        output_format='WAV',
        normalization_threshold=0.9,
        output_single_stem='Vocals',  # Выделяем только вокал
        use_autocast=torch.cuda.is_available(),  # Используем autocast только если есть GPU
    )
    
    # Загружаем модель один раз (используем высококачественную модель для вокала)
    try:
        separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
        logger.info("Loaded BS-Roformer model for vocal separation")
    except Exception as e:
        logger.warn(f"Failed to load BS-Roformer model, trying default: {e}")
        try:
            separator.load_model()  # Загружаем модель по умолчанию
            logger.info("Loaded default model for vocal separation")
        except Exception as e:
            logger.error(f"Failed to load any model: {e}")
            return

    for song_dir in song_dirs:
        index = int(song_dir.name)
        
        # Проверяем, выполнена ли уже задача
        if not overwrite and check_task_done(TASK_VOCAL, output_dir / song_dir_name(index)):
            logger.info(f"Vocal extraction for song {index} already done. Skip.")
            continue

        song_path = song_dir / "song.mp3"
        out_song_dir = output_dir / song_dir_name(index)
        out_song_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверяем существование исходного файла
        if not song_path.exists():
            logger.warn(f"song.mp3 not found in {song_dir}, skipping.")
            continue

        vocal_path = out_song_dir / "sep_vocals.wav"
        
        # Проверяем, существует ли уже файл с вокалом
        if not overwrite and vocal_path.exists():
            logger.info(f"Vocal file already exists for song {index}, skipping.")
            continue

        logger.info(f"Extracting vocal from {song_path} to {vocal_path}")
        
        try:
            # Временно устанавливаем выходную директорию для separator
            original_output_dir = separator.output_dir
            separator.output_dir = str(out_song_dir)
            
            # Выполняем разделение
            output_files = separator.separate(str(song_path))
            
            # Восстанавливаем оригинальную выходную директорию
            separator.output_dir = original_output_dir
            
            # Находим файл с вокалом среди выходных файлов и переименовываем его
            for output_file in output_files:
                output_path = Path(output_file)
                if 'Vocals' in output_path.name or 'vocals' in output_path.name.lower():
                    if output_path != vocal_path:
                        output_path.rename(vocal_path)
                    break
            
            if vocal_path.exists():
                logger.info(f"Successfully extracted vocal to {vocal_path}")
            else:
                logger.error(f"Vocal file was not created for {song_path}")
                continue
                
        except Exception as e:
            logger.error(f"Failed to extract vocal from {song_path}: {e}")
            continue

    # Отмечаем задачу как выполненную после обработки всех файлов
    for song_dir in song_dirs:
        index = int(song_dir.name)
        out_song_dir = output_dir / song_dir_name(index)
        vocal_path = out_song_dir / "sep_vocals.wav"
        
        # Отмечаем задачу выполненной только если файл с вокалом существует
        if vocal_path.exists():
            mark_task_done(TASK_VOCAL, output_dir / song_dir_name(index))


def get_highest_pitches_from_piano_roll(pr):
    """
    params:
        pr : (128, time(frame))

    return:
        highest_pitches : (time(frame), )
    """
    highest_pitches = []
    for i in range(pr.shape[1]):
        ps = np.nonzero(pr[:, i])
        if len(ps[0]) == 0:
            highest_pitches.append(np.nan)
        else:
            highest_pitches.append(ps[0][-1])
    highest_pitches = np.array(highest_pitches)

    return highest_pitches


def compute_melody_chroma_accuracy(
    data_dir: Path,
    output_dir: Path,
    partition: Tuple[int, int],
    debug: bool = False,
    overwrite: bool = False,
):
    overwrite = True
    song_dirs = _get_song_dirs(data_dir, partition, debug)
    for song_dir in song_dirs:
        index = int(song_dir.name)
        if not overwrite and check_task_done(TASK_MEL_CHROMA, output_dir / song_dir_name(index)):
            logger.warn(f"Task {index} has been done. Skip.")
            continue

        logger.info("Compute melody chroma accuracy for song %d", index)
        (output_dir / song_dir_name(index)).mkdir(parents=True, exist_ok=True)

        piano_midi_file = output_dir / song_dir_name(index) / "piano.mid"
        if not piano_midi_file.exists():
            logger.warn(f"Piano MIDI file {piano_midi_file} does not exist, skipping.")
            continue

        vocals_file = output_dir / song_dir_name(index) / "sep_vocals.wav"
        if not vocals_file.exists():
            logger.warn(f"Vocals file {vocals_file} does not exist, skipping.")
            continue

        try:
            song_midi = pretty_midi.PrettyMIDI(str(piano_midi_file))
            vocals = librosa.load(str(vocals_file), sr=44100)[0]
            raw_chroma = melody_chroma_accuracy(song_midi, vocals)
            
            logger.info(f"Melody chroma accuracy for song {index}: {raw_chroma}")
            # Сохраняем метрику в txt-файл
            mca_file = output_dir / song_dir_name(index) / "mca.txt"
            with open(mca_file, "w") as f:
                f.write(f"{raw_chroma}\n")
        except Exception as e:
            logger.error(f"Failed to compute melody chroma accuracy for song {index}: {e}")
            continue

        mark_task_done(TASK_MEL_CHROMA, output_dir / song_dir_name(index))


def visualize_melody_chroma(
    data_dir: Path,
    output_dir: Path,
    partition: Tuple[int, int],
    debug: bool = False,
    overwrite: bool = False,
):
    song_dirs = _get_song_dirs(data_dir, partition, debug)
    mca_values = []
    for song_dir in song_dirs:
        index = int(song_dir.name)
        mca_file = output_dir / song_dir_name(index) / "mca.txt"
        if mca_file.exists():
            try:
                with open(mca_file, "r") as f:
                    value = float(f.read().strip())
                    mca_values.append(value)
            except Exception as e:
                logger.warn(f"Could not read mca.txt for song {index}: {e}")
                continue
    if not mca_values:
        logger.warn("No MCA values found for visualization.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(mca_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('Melody Chroma Accuracy Distribution')
    plt.xlabel('MCA')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = output_dir / "mca_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved MCA distribution plot to {plot_path}")
