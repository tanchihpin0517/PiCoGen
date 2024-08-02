import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import ly.document
import ly.music
import numpy as np
import pretty_midi as pm
import torch
import validators
from tqdm import tqdm

from .model import CPTransformer
from .repr import DEFAULT_BPM_BINS, LS_DEFAULT_VELOCITY, Event, Tokenizer
from .utils import load_checkpoint, load_config, query_mkdir

LEADSHEET_DIR_NAME = "leadsheet"


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="stage", required=True)

    # Stage 1 parser
    stage1_parser = subparsers.add_parser("stage1")
    stage1_parser.add_argument("--input_url_or_file", type=str)
    stage1_parser.add_argument("--output_dir", type=Path, required=True)

    # Stage 2 parser
    stage2_parser = subparsers.add_parser("stage2")
    stage2_parser.add_argument("--leadsheet_dir", type=Path, required=True)
    stage2_parser.add_argument("--output_dir", type=Path, required=True)
    stage2_parser.add_argument("--config_file", type=Path, required=True)
    stage2_parser.add_argument("--ckpt_file", type=Path, required=True)
    stage2_parser.add_argument("--vocab_file", type=Path, required=True)

    return parser.parse_args()


def main():
    ca = parse_args()
    query_mkdir(ca.output_dir)

    if ca.stage == "stage1":
        run_sheetsage(ca)
    elif ca.stage == "stage2":
        gen_piano_cover(ca)
    else:
        raise ValueError(f"Unknown stage: {ca.stage}")


def run_sheetsage(ca):
    if validators.url(ca.input_url_or_file):
        input_file = download_audio(ca.input_url_or_file, ca.output_dir)
    else:
        input_file = ca.output_dir / "song.mp3"
        shutil.copy(ca.input_url_or_file, input_file)

    extract_leadsheet(input_file, ca.output_dir)


def download_audio(url, output_dir):
    output_file = output_dir / "song.mp3"
    if output_file.exists():
        output_file.unlink()
    cmd = f"yt-dlp --extract-audio --audio-format mp3 {url} -o '{output_dir}/song.%(ext)s'"
    subprocess.run(cmd, shell=True, check=True)
    return output_file


def extract_leadsheet(audio_file, output_dir):
    tmp_dir = tempfile.TemporaryDirectory()

    try:
        cmd = f"python -m sheetsage.sheetsage.infer -j --output_dir {tmp_dir.name} {audio_file}"
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        cmd = f"python -m sheetsage.sheetsage.infer -j --output_dir {tmp_dir.name} {audio_file} --measures_per_chunk 4"
        subprocess.run(cmd, shell=True, check=True)

    ls_dir = output_dir / "leadsheet"
    ls_dir.mkdir(exist_ok=True)
    for file in Path(tmp_dir.name).glob("*"):
        dst = ls_dir / file.name
        shutil.copy(file, dst)


@torch.no_grad()
def gen_piano_cover(ca):
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(0))
    else:
        device = torch.device("cpu")

    hp = load_config(ca.config_file, verbose=True)
    model = CPTransformer(hp).to(device)
    state_dict = load_checkpoint(ca.ckpt_file, device)
    model.load_state_dict(state_dict["model"])

    model.eval()

    tokenizer = Tokenizer(ca.vocab_file, hp.beat_div, hp.ticks_per_beat)

    print(f"Lead Sheet: {ca.leadsheet_dir}")

    try:
        ls_file = ca.leadsheet_dir / "output.ly"
        bpm, _, chord_quals = get_tempo_and_cquals(ls_file)
        ls_events, beat_per_bar = get_cp_with_empty_chord(
            ca.leadsheet_dir / "output.midi",
            tokenizer.beat_div,
            tempo=bpm,
            cquals=chord_quals,
        )
    except ValueError as e:
        print(f"Error processing lead sheet of {ca.leadsheet_dir}: {e}")
        exit(1)

    bos_event = Event(family="spec", spec="bos")
    end_event = Event(family="spec", spec="se")
    bar_src_event = Event(family="bar", bar="src")

    ls_ids = [tokenizer.e2i(e) for e in ls_events]
    tgt_seg = [tokenizer.e2i(bos_event)]
    family_mask = tokenizer.get_family_mask().to(device)

    last_past_kv = None
    out_events = []

    bar_ranges = tokenizer.get_bar_ranges(ls_events)
    pbar = tqdm(total=len(bar_ranges))
    for ls_bar_start, ls_bar_end in bar_ranges:
        tgt_seg.extend(ls_ids[ls_bar_start:ls_bar_end])
        if ls_bar_start == 0:
            tgt_seg.append(tokenizer.e2i(Event(family="spec", spec="ss")))

        bar_tgt_event = Event(family="bar", bar="tgt")
        tgt_seg.append(tokenizer.e2i(bar_tgt_event))

        while True:
            if len(tgt_seg) >= 1024:
                tgt_seg = tgt_seg[512:]
                tgt_seg = [tokenizer.e2i(bos_event)] + tgt_seg

            input_ids = torch.LongTensor(tgt_seg)[None, :, :].to(device)
            output_ids, _ = model.generate(input_ids, family_mask, last_past_kv)
            out_id = output_ids[0][-1].tolist()
            out_event = tokenizer.i2e(out_id)

            out_events.append(out_event)

            if out_event in (bar_src_event, end_event):
                break
            # if len(out_events) > 10240:
            #     break

            tgt_seg.append(out_id)
            pbar.set_description(f"length: {len(tgt_seg)}")

        if len(out_events) > 10240:
            break

        pbar.update(1)
    pbar.close()

    piano_output_file = ca.output_dir / "cover.mid"
    tokenizer.events_to_midi(
        out_events,
        beat_per_bar,
    ).dump(piano_output_file)
    (ca.output_dir / "cover.txt").write_text("\n".join([str(e) for e in out_events]))


def _get_item_content(item):
    return str(item).split("'")[1].split("'")[0]


def get_tempo_and_cquals(lily_file):
    doc = ly.music.document(ly.document.Document(lily_file.read_text()))

    beat_per_bar = doc[1][1][0][0][2].numerator()
    ts_den = 1 / doc[1][1][0][0][2].fraction()
    assert ts_den == 4

    tempo_dur = doc[1][1][0][0][3][0]  # ly.music.items.Duration
    tempo_num = doc[1][1][0][0][3][1]  # ly.music.items.Number
    dur = int(_get_item_content(tempo_dur))
    assert dur == 4
    bpm = int(tempo_num.value() * (1 / 4 / (1 / dur)))

    chord_list = doc[1][0][0][2][0]
    chord_quals = []
    qual_map = {
        "m": "m",
        "m7": "m7",
        "maj7": "M7",
        "7": "7",
        "dim": "o",
        "sus4": "sus4",
    }
    for item in chord_list:
        if type(item) == ly.music.items.Note:
            chord_quals.append("M")
        elif type(item) == ly.music.items.ChordSpecifier:
            qual = "".join([_get_item_content(i) for i in item[1:]])
            assert qual in qual_map, f"Unknown chord quality: {qual}"
            chord_quals[-1] = qual_map[qual]

    return bpm, beat_per_bar, chord_quals


def get_cp_with_empty_chord(midi_file, beat_div, tempo=120, cquals=[]):
    midi = pm.PrettyMIDI(str(midi_file))
    drum, chord, melody = midi.instruments

    beat_per_bar = 1
    for n in drum.notes[1:]:
        if n.pitch == drum.notes[0].pitch:
            break
        beat_per_bar += 1

    subbeat_bins = []
    for n in drum.notes:
        for i in range(beat_div):
            subbeat_bins.append(n.start + (n.end - n.start) / beat_div * i)
    subbeat_bins = np.array(subbeat_bins)

    melody_grid = [list() for _ in subbeat_bins]
    chord_note_grid = [list() for _ in subbeat_bins]

    for note in melody.notes:
        onset = np.argmin(np.abs(subbeat_bins - note.start))
        offset = np.argmin(np.abs(subbeat_bins - note.end))
        melody_grid[onset].append((note.pitch, note.velocity, offset - onset))

    for note in chord.notes:
        onset = np.argmin(np.abs(subbeat_bins - note.start))
        offset = np.argmin(np.abs(subbeat_bins - note.end))
        chord_note_grid[onset].append(note.pitch)

    bpm = DEFAULT_BPM_BINS[np.argmin(np.abs(DEFAULT_BPM_BINS - tempo))]

    events = [Event(family="spec", spec="ss")]
    subbeat_per_bar = beat_div * beat_per_bar
    root_map = {
        0: "A",
        1: "A#",
        2: "B",
        3: "C",
        4: "C#",
        5: "D",
        6: "D#",
        7: "E",
        8: "F",
        9: "F#",
        10: "G",
        11: "G#",
    }
    count = 0
    for i in range(len(melody_grid)):
        if i % subbeat_per_bar == 0:
            events.append(Event(family="bar", bar="src"))

        if i == 0:
            events.append(
                Event(family="metric", position=str(0), chord="N_N", tempo=bpm)
            )

        if len(chord_note_grid[i]) > 0 or len(melody_grid[i]) > 0:
            if i != 0:
                events.append(
                    Event(
                        family="metric",
                        position=i % subbeat_per_bar,
                        chord="cont",
                        tempo="cont",
                    )
                )

        if len(chord_note_grid[i]) > 0:
            root = root_map[(sorted(chord_note_grid[i])[0] - 21) % 12]
            events[-1].chord = f"{root}_{cquals[count]}"

            count += 1

        if len(melody_grid[i]) > 0:
            for pitch, velocity, duration in melody_grid[i]:
                events.append(
                    Event(
                        family="note",
                        pitch=pitch,
                        velocity=LS_DEFAULT_VELOCITY,
                        duration=max(min(duration, 16), 1),
                    )
                )

    assert count == len(cquals)

    if events[-1].family == "bar":
        events.pop()

    events.append(Event(family="spec", spec="se"))

    return events, beat_per_bar


if __name__ == "__main__":
    main()
