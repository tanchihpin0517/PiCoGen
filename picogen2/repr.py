import collections
import json
from itertools import chain
from pathlib import Path

import miditoolkit
import numpy as np

from . import assets
from .utils import load_config

DEFAULT_SUBBEAT_RANGE = np.arange(0, 64, dtype=int)
DEFAULT_PIANO_RANGE = np.arange(21, 109, dtype=int)
DEFAULT_VELOCITY_BINS = np.linspace(0, 124, 31 + 1, dtype=int)  # midi velocity: 0~127
LS_DEFAULT_VELOCITY = 80
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
DEFAULT_DURATION_RANGE = np.arange(1, 1 + 32, dtype=int)
DEFAULT_CHORD_ROOTS = [
    "A",
    "A#",
    "B",
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
]
DEFAULT_CHORD_QUALITY = [
    "+",
    "/o7",
    "7",
    "M",
    "M7",
    "m",
    "m7",
    "o",
    "o7",
    "sus2",
    "sus4",
]
VOCAB_SIZE = 500


def gen_vocab():
    spec = [f"spec_{t}" for t in ["pad", "bos", "eos", "unk", "mask", "ss", "se"]]
    bar = ["bar_start", "bar_end"] + [f"bar_{i}" for i in range(1, 5)] + ["bar_N"]
    position = [f"position_{i}" for i in DEFAULT_SUBBEAT_RANGE]
    chord = ["chord_N_N"]
    for root in DEFAULT_CHORD_ROOTS:
        for quality in DEFAULT_CHORD_QUALITY:
            chord.append(f"chord_{root}_{quality}")
    tempo = [f"tempo_{i}" for i in DEFAULT_BPM_BINS]
    pitch = [f"pitch_{i}" for i in DEFAULT_PIANO_RANGE]
    duration = [f"duration_{i}" for i in DEFAULT_DURATION_RANGE]
    velocity = [f"velocity_{i}" for i in DEFAULT_VELOCITY_BINS]
    vocab = spec + bar + position + chord + tempo + pitch + duration + velocity
    vocab = vocab + ["reserved"] * (VOCAB_SIZE - len(vocab))
    return vocab


class Event:
    def __init__(
        self,
        etype,
        value,
    ):
        self.etype = etype
        self.value = value

    def init_check(self):
        if self.etype == "spec":
            assert self.value.split("_")[0] in ["spec"], f"{self.etype}: {self.value}"

        elif self.etype == "bar":
            assert self.value.split("_")[0] in ["bar"], f"{self.etype}: {self.value}"

        elif self.etype == "metric":
            assert self.value.split("_")[0] in [
                "position",
                "chord",
                "tempo",
            ], f"{self.etype}: {self.value}"

        elif self.etype == "note":
            assert self.value.split("_")[0] in [
                "pitch",
                "duration",
                "velocity",
            ], f"{self.etype}: {self.value}"

        else:
            raise ValueError(f"Unknown etype: {self.etype}")

    def unwrap(self, vtype):
        return vtype(self.value.split("_")[1])

    def __repr__(self):
        return f"Event({self.etype}: {self.value})"

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.etype == other.etype and self.value == other.value


class Tokenizer:
    def __init__(self, vocab_file=None, beat_div=None, ticks_per_beat=None):
        vocab_file = assets.vocab_file() if vocab_file is None else vocab_file

        if beat_div is None or ticks_per_beat is None:
            config_file = assets.config_file()
            hp = load_config(config_file)
            beat_div = hp.beat_div
            ticks_per_beat = hp.ticks_per_beat

        self.vocab = Vocab(vocab_file)
        self.beat_div = beat_div
        self.ticks_per_beat = ticks_per_beat

    def get_song_from_midi(self, midi):
        song = midi_to_song(midi, self.beat_div)
        song["events"] = song_to_events(song)

        song["ls_events"] = extract_leadsheet_from_events(
            song["events"], song["metadata"]["beat_per_bar"], self.beat_div
        )

        return song

    def events_to_midi(self, events):
        midi = events_to_midi(events, self.ticks_per_beat, self.beat_div)
        return midi

    def e2i(self, e):
        return self.vocab.t2i[e.value]

    def i2e(self, eid):
        token = self.vocab.i2t[eid]
        if token.startswith("spec"):
            return Event("spec", token)
        elif token.startswith("bar"):
            return Event("bar", token)
        elif token.split("_")[0] in ["position", "chord", "tempo"]:
            return Event("metric", token)
        elif token.split("_")[0] in ["pitch", "duration", "velocity"]:
            return Event("note", token)
        else:
            raise ValueError(f"Unknown token: {token}")

    def get_bar_ranges(self, events, from_start=True):
        return get_bar_ranges(events, from_start)

    @staticmethod
    def get_tempo_event(bpm):
        tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - bpm))]
        return Event("metric", f"tempo_{tempo}")

    @staticmethod
    def get_duration_event(duration):
        duration = DEFAULT_DURATION_RANGE[np.argmin(abs(DEFAULT_DURATION_RANGE - duration))]
        return Event("note", f"duration_{duration}")

    @staticmethod
    def get_velocity_event(velocity):
        velocity = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS - velocity))]
        return Event("note", f"velocity_{velocity}")


class Vocab:
    def __init__(self, vacab_file):
        self.i2t = json.loads(vacab_file.read_text())
        self.t2i = {}
        for i, t in enumerate(self.i2t):
            self.t2i[t] = i

    def len(self):
        return len(self.i2t)

    def __len__(self):
        return len(self.i2t)

    def __repr__(self):
        out = []
        for i, t in enumerate(self.i2t):
            out.append(f"{i}: {t}")
        return "\n".join(out)


def midi_to_song(midi_obj, beat_div):
    assert midi_obj.ticks_per_beat % beat_div == 0
    grid_resol = midi_obj.ticks_per_beat // beat_div

    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        for note in instr.notes:
            instr_notes[instr.name].append(note)
        instr_notes[instr.name].sort(key=lambda x: x.start)

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split("_")[0] != "global" and "Boundary" not in marker.text.split("_")[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if "Boundary" in marker.text.split("_")[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    global_bpm = None
    for marker in midi_obj.markers:
        if marker.text.split("_")[0] == "global" and marker.text.split("_")[1] == "bpm":
            global_bpm = int(marker.text.split("_")[2])

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            # quantize start
            quant_time = round(note.start / grid_resol)

            # duration
            note_duration = note.end - note.start
            duration = round(note_duration / grid_resol)
            duration = max(duration, 1)  # dur >= 1

            # append
            note_grid[quant_time].append(
                {
                    "note": note,
                    "pitch": note.pitch,
                    "duration": duration,
                    "velocity": note.velocity,
                }
            )

        # sort
        for time in note_grid.keys():
            note_grid[time].sort(key=lambda x: -x["pitch"])

        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        quant_time = round(chord.time / grid_resol)
        # chord_grid[quant_time] = [chord] # NOTE: only one chord per time
        chord_grid[quant_time].append(chord)

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        quant_time = round(tempo.time / grid_resol)
        # tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo.tempo))]
        tempo_grid[quant_time] = [tempo]  # NOTE: only one tempo per time

    all_bpm = [tempo[0].tempo for _, tempo in tempo_grid.items()]
    assert len(all_bpm) > 0, " No tempo changes in midi file."
    average_bpm = sum(all_bpm) / len(all_bpm)
    if global_bpm is None:
        global_bpm = average_bpm

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        quant_time = round(label.time / grid_resol)
        label_grid[quant_time] = [label]

    # collect
    song_data = {
        "notes": intsr_gird,
        "chords": chord_grid,
        "tempos": tempo_grid,
        "labels": label_grid,
        "metadata": {
            "global_bpm": global_bpm,
            "average_bpm": average_bpm,
            "beat_div": beat_div,
            "beat_per_bar": midi_obj.time_signature_changes[0].numerator,
        },
    }
    return song_data


def song_to_events(song):
    beat_div = song["metadata"]["beat_div"]
    beat_per_bar = song["metadata"]["beat_per_bar"]
    grid_per_bar = beat_div * beat_per_bar

    events = [Event("spec", "spec_ss")]
    global_tempo = DEFAULT_BPM_BINS[
        np.argmin(abs(DEFAULT_BPM_BINS - song["metadata"]["global_bpm"]))
    ]
    events.append(Event("metric", f"tempo_{global_tempo}"))

    max_grid = list(chain(song["tempos"].keys(), song["chords"].keys()))
    for _, v in song["notes"].items():
        max_grid.extend(v.keys())
    max_grid = max(max_grid)

    for bar_i in range(0, max_grid + 1, grid_per_bar):
        events.append(Event("bar", "bar_start"))
        for i in range(bar_i, min(bar_i + grid_per_bar, max_grid + 1)):
            pos = Event("metric", f"position_{i-bar_i}")
            tmp = []
            empty = True
            if i in song["chords"]:
                chord_items = song["chords"][i][0].text.split("_")
                chord = f"{chord_items[0]}_{chord_items[1]}"
                tmp.append(Event("metric", f"chord_{chord}"))
                empty = False
            if i in song["tempos"]:
                tempo = DEFAULT_BPM_BINS[
                    np.argmin(abs(DEFAULT_BPM_BINS - song["tempos"][i][0].tempo))
                ]
                tmp.append(Event("metric", f"tempo_{tempo}"))
                empty = False
            for _, instr in song["notes"].items():
                if i in instr:
                    for note in instr[i]:
                        duration = DEFAULT_DURATION_RANGE[
                            np.argmin(abs(DEFAULT_DURATION_RANGE - note["duration"]))
                        ]
                        velocity = DEFAULT_VELOCITY_BINS[
                            np.argmin(abs(DEFAULT_VELOCITY_BINS - note["velocity"]))
                        ]
                        tmp.append(Event("note", f'pitch_{note["pitch"]}'))
                        tmp.append(Event("note", f"duration_{duration}"))
                        tmp.append(Event("note", f"velocity_{velocity}"))
                        empty = False

            if not empty:
                events.append(pos)
            events.extend(tmp)
        events.append(Event("bar", "bar_end"))

    events.append(Event("spec", "spec_se"))

    return events


def events_to_midi(events, ticks_per_beat, grid_div):
    bar_ranges = get_bar_ranges(events, from_start=False)

    midi = miditoolkit.MidiFile()
    midi.ticks_per_beat = ticks_per_beat
    track = miditoolkit.Instrument(program=0, is_drum=False, name="piano")
    midi.instruments = [track]

    bar_tick = 0
    subbeat_tick = 0

    pitch, velocity, duration = 0, 0, 0
    bar_len = 4

    for i, (start, end) in enumerate(bar_ranges):
        assert ticks_per_beat % grid_div == 0
        ticks_per_subbeat = ticks_per_beat // grid_div

        for event in events[start:end]:
            if event.etype == "spec":
                pass
            elif event.etype == "bar":
                if event.value in ["bar_start", "bar_end"]:
                    pass
                else:
                    try:
                        bar_len = int(event.value.split("_")[1])
                    except ValueError:
                        assert event.value == "bar_N"

            elif event.etype == "metric":
                v = event.value
                if v.startswith("position"):
                    pos = int(v.split("_")[1])
                    subbeat_tick = pos * ticks_per_subbeat
                elif v.startswith("tempo"):
                    tempo = int(v.split("_")[1])
                    m = miditoolkit.TempoChange(time=bar_tick + subbeat_tick, tempo=tempo)
                    midi.tempo_changes.append(m)
                elif v.startswith("chord"):
                    pass
                else:
                    raise ValueError(f"Unknown metric: {v}")
            elif event.etype == "note":
                v = event.value
                if v.startswith("pitch"):
                    pitch = int(v.split("_")[1])
                elif v.startswith("duration"):
                    duration = int(v.split("_")[1]) * ticks_per_subbeat
                elif v.startswith("velocity"):
                    velocity = int(v.split("_")[1])
                    n = miditoolkit.Note(
                        start=bar_tick + subbeat_tick,
                        end=bar_tick + subbeat_tick + duration,
                        pitch=pitch,
                        velocity=velocity,
                    )
                    midi.instruments[0].notes.append(n)
                else:
                    raise ValueError(f"Unknown note: {v}")
            else:
                raise ValueError(f"Unknown event: {type(event)}")

        bar_tick += ticks_per_beat * bar_len

    return midi


def extract_leadsheet_from_events(
    events, beat_per_bar, beat_div, cover_beat=0, min_pitch=60, no_chord=False
):
    # algorithm:
    # - skyline
    # - filter out notes with pitch < 60

    grids = []
    for event in events:
        if event.etype == "bar":
            for _ in range(beat_per_bar * beat_div):
                grids.append(list())

    grid_idx = 0
    bar_count = 0
    subbeat = 0
    note_tmp = []
    first_tempo = True
    for event in events:
        if event.etype == "spec":
            grids[grid_idx].append(event)
        elif event.etype == "bar":
            if event.value == "bar_end":
                bar_count += 1
                grid_idx = bar_count * (beat_per_bar * beat_div)
                subbeat = 0
            grids[grid_idx].append(event)
        elif event.etype == "metric":
            if event.value.startswith("position"):
                subbeat = int(event.value.split("_")[1])
            if not event.value.startswith("tempo"):
                grids[grid_idx + subbeat].append(event)
            else:  # tempo
                if first_tempo:
                    grids[grid_idx + subbeat].append(event)
                    first_tempo = False
        elif event.etype == "note":
            note_tmp.append(event)
            if event.value.startswith("velocity"):
                pitch = note_tmp[0].value.split("_")[1]
                if int(pitch) >= min_pitch:  # only keep notes with pitch >= min_pitch
                    grids[grid_idx + subbeat].append(note_tmp[0])
                    grids[grid_idx + subbeat].append(note_tmp[1])
                    grids[grid_idx + subbeat].append(
                        Event("note", f"velocity_{LS_DEFAULT_VELOCITY}")
                    )
                note_tmp = []

    # select the highest note
    for i, grid in enumerate(grids):
        notes = [e for e in grid if e.etype == "note"]
        notes = [(notes[i], notes[i + 1], notes[i + 2]) for i in range(0, len(notes), 3)]
        notes.sort(key=lambda x: -x[0].unwrap(int))  # sort by pitch
        grids[i] = [e for e in grid if not e.etype == "note"]
        if len(notes) > 0:
            grids[i].extend(notes[0])

    # remove useless metric
    for i, grid in enumerate(grids):
        if len(grid) == 0 or not grid[-1].etype == "metric":
            continue
        metric = grid[-1]
        assert metric.etype == "metric"
        if metric.value.startswith("position"):
            grids[i].pop()
            assert len(grid) == 0 or grid[-1].etype == "bar"

    return list(chain(*grids))


def get_bar_ranges(events, from_start=True):
    bar_idx_list = []
    for i, event in enumerate(events):
        if event.etype == "bar" and event.value == "bar_start":
            bar_idx_list.append(i)
    bar_idx_list = bar_idx_list + [len(events)]
    if from_start:
        bar_idx_list[0] = 0  # the first bar starts at 0

    bar_ranges = []
    for i in range(len(bar_idx_list) - 1):
        bar_ranges.append((bar_idx_list[i], bar_idx_list[i + 1]))
    return bar_ranges


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cmd_gen_vocab = subparsers.add_parser("gen_vocab")
    cmd_gen_vocab.add_argument("--output_file", type=Path, required=True)
    ca = parser.parse_args()

    if ca.command is None:
        parser.print_help()
        exit()
    elif ca.command == "gen_vocab":
        vocab = gen_vocab()
        ca.output_file.write_text(json.dumps(vocab, indent=2))
        vocab = Vocab(ca.output_file)
        print(vocab)
        print("vocab size:", vocab.len())
    else:
        raise ValueError(f"Unknown command: {ca.command}")
