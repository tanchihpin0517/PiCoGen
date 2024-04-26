from itertools import chain
from pathlib import Path
import miditoolkit
import collections
import numpy as np
import json
from collections import UserList
import torch
import torch.nn.functional as F
from copy import deepcopy
# from itertools import pairwise

DEFAULT_SUBBEAT_RANGE = np.arange(0, 32, dtype=int)
DEFAULT_PIANO_RANGE = np.arange(21, 109, dtype=int)
DEFAULT_VELOCITY_BINS = np.linspace(0,  124, 31+1, dtype=int) # midi velocity: 0~127
LS_DEFAULT_VELOCITY = 80
DEFAULT_BPM_BINS = np.linspace(32, 224, 64+1, dtype=int)
DEFAULT_DURATION_RANGE = np.arange(1, 1+32, dtype=int)
DEFAULT_CHORD_ROOTS = [
    "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
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

def gen_vocab():
    spec = [f'spec_{t}' for t in ['bos', 'eos', 'unk', 'mask', 'ss', 'se']]
    family = [f'family_{t}' for t in ['spec', 'bar', 'metric', 'note']]
    bar = ['bar_src', 'bar_tgt']
    position = [f'position_{i}' for i in DEFAULT_SUBBEAT_RANGE]
    chord = ['chord_cont', 'chord_N_N']
    for root in DEFAULT_CHORD_ROOTS:
        for quality in DEFAULT_CHORD_QUALITY:
            chord.append(f"chord_{root}_{quality}")
    tempo = ['tempo_cont'] + [f'tempo_{i}' for i in DEFAULT_BPM_BINS]
    pitch = [f'pitch_{i}' for i in DEFAULT_PIANO_RANGE]
    duration = [f'duration_{i}' for i in DEFAULT_DURATION_RANGE]
    velocity = [f'velocity_{i}' for i in DEFAULT_VELOCITY_BINS]
    vocab = ['ign', {
        'spec': spec,
        'family': family,
        'bar': bar,
        'position': position,
        'chord': chord,
        'tempo': tempo,
        'pitch': pitch,
        'duration': duration,
        'velocity': velocity,
    }]
    return vocab

def gen_vocab_midi():
    vocab = ['bos', 'eos', 'unk', 'mask', 'ss', 'se']
    vocab += ['bar_src', 'bar_tgt']
    vocab += ['note_on', 'note_off']
    vocab += [f'pitch_{i}' for i in range(21, 109)]
    vocab += [f'beat_{i}' for i in range(32)]

    chord = ['chord_N_N']
    for root in DEFAULT_CHORD_ROOTS:
        for quality in DEFAULT_CHORD_QUALITY:
            chord.append(f"chord_{root}_{quality}")
    vocab += chord

    velocity = [f'velocity_{i}' for i in DEFAULT_VELOCITY_BINS]
    vocab += velocity

    tempo = ['tempo_cont'] + [f'tempo_{i}' for i in DEFAULT_BPM_BINS]
    vocab += tempo

    vocab = [f'spec_{t}' for t in vocab]
    vocab = ['ign'] + vocab

    return vocab

class Event(UserList): # CPWord
    WORD_SIZE = 9

    IDX_FAMILY = 0
    IDX_SPEC = 1
    IDX_BAR = 2
    IDX_POSITION = 3
    IDX_CHORD = 4
    IDX_TEMPO = 5
    IDX_PITCH = 6
    IDX_DURATION = 7
    IDX_VELOCITY = 8

    def __init__(
        self,
        family,
        spec = 'ign',
        bar = 'ign',
        position = 'ign',
        chord = 'ign',
        tempo = 'ign',
        pitch = 'ign',
        duration = 'ign',
        velocity = 'ign',
    ):
        # family, spec, bar, position, chord, tempo, pitch, duration, velocity
        # 0       1     2    3         4      5      6      7         8
        self.data = [None] * self.WORD_SIZE
        self.family = family
        self.spec = spec
        self.bar = bar
        self.position = position
        self.chord = chord
        self.tempo = tempo
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity

        self.init_check()

    def init_check(self):
        if self.family == 'spec':
            val_idx = [self.IDX_FAMILY, self.IDX_SPEC]
            ign_idx = [i for i in range(self.WORD_SIZE) if i not in val_idx]
            assert all([not self.data[i].endswith('ign') for i in val_idx]), f'{self.data}'
            assert all([self.data[i].endswith('ign') for i in ign_idx]), f'{self.data}'

        elif self.family == 'bar':
            val_idx = [self.IDX_FAMILY, self.IDX_BAR]
            ign_idx = [i for i in range(self.WORD_SIZE) if i not in val_idx]
            assert all([not self.data[i].endswith('ign') for i in val_idx]), f'{self.data}'
            assert all([self.data[i].endswith('ign') for i in ign_idx]), f'{self.data}'

        elif self.family == 'metric':
            val_idx = [self.IDX_FAMILY, self.IDX_POSITION, self.IDX_CHORD, self.IDX_TEMPO]
            ign_idx = [i for i in range(self.WORD_SIZE) if i not in val_idx]
            assert all([not self.data[i].endswith('ign') for i in val_idx]), f'{self.data}'
            assert all([self.data[i].endswith('ign') for i in ign_idx]), f'{self.data}'

        elif self.family == 'note':
            val_idx = [self.IDX_FAMILY, self.IDX_PITCH, self.IDX_DURATION, self.IDX_VELOCITY]
            ign_idx = [i for i in range(self.WORD_SIZE) if i not in val_idx]
            assert all([not self.data[i].endswith('ign') for i in val_idx]), f'{self.data}'
            assert all([self.data[i].endswith('ign') for i in ign_idx]), f'{self.data}'

        else:
            raise ValueError(f'Unknown family: {self.data}')

    def __repr__(self):
        non_ign = []
        for item in self.data[1:]:
            if item.split('_', 1)[-1] != 'ign':
                non_ign.append(item)
        f = self.data[self.IDX_FAMILY].split('_', 1)[-1]
        return f'Event({f}: {non_ign})'

    @property
    def family(self):
        return self.data[self.IDX_FAMILY].split('_', 1)[-1]

    @family.setter
    def family(self, v):
        self.data[self.IDX_FAMILY] = f'family_{v}'

    @property
    def spec(self):
        return self.data[self.IDX_SPEC].split('_', 1)[-1]

    @spec.setter
    def spec(self, v):
        self.data[self.IDX_SPEC] = f'spec_{v}' if v != 'ign' else 'ign'

    @property
    def bar(self):
        return self.data[self.IDX_BAR].split('_', 1)[-1]

    @bar.setter
    def bar(self, v):
        self.data[self.IDX_BAR] = f'bar_{v}' if v != 'ign' else 'ign'

    @property
    def position(self):
        return int(self.data[self.IDX_POSITION].split('_', 1)[-1])

    @position.setter
    def position(self, v):
        self.data[self.IDX_POSITION] = f'position_{v}' if v != 'ign' else 'ign'

    @property
    def chord(self):
        return self.data[self.IDX_CHORD].split('_', 1)[-1]

    @chord.setter
    def chord(self, v):
        self.data[self.IDX_CHORD] = f'chord_{v}' if v != 'ign' else 'ign'

    @property
    def tempo(self):
        try:
            return int(self.data[self.IDX_TEMPO].split('_', 1)[-1])
        except ValueError:
            return self.data[self.IDX_TEMPO].split('_', 1)[-1]

    @tempo.setter
    def tempo(self, v):
        self.data[self.IDX_TEMPO] = f'tempo_{v}' if v != 'ign' else 'ign'

    @property
    def pitch(self):
        return int(self.data[self.IDX_PITCH].split('_', 1)[-1])

    @pitch.setter
    def pitch(self, v):
        self.data[self.IDX_PITCH] = f'pitch_{v}' if v != 'ign' else 'ign'

    @property
    def duration(self):
        return int(self.data[self.IDX_DURATION].split('_', 1)[-1])

    @duration.setter
    def duration(self, v):
        self.data[self.IDX_DURATION] = f'duration_{v}' if v != 'ign' else 'ign'

    @property
    def velocity(self):
        return int(self.data[self.IDX_VELOCITY].split('_', 1)[-1])

    @velocity.setter
    def velocity(self, v):
        self.data[self.IDX_VELOCITY] = f'velocity_{v}' if v != 'ign' else 'ign'

class Tokenizer:
    def __init__(self, vocab_file, beat_div, ticks_per_beat):
        self.vocab = Vocab(vocab_file)
        self.beat_div = beat_div
        self.ticks_per_beat = ticks_per_beat

    def get_song_from_midi(self, midi):
        # midi = miditoolkit.midi.parser.MidiFile(midi)
        song = midi_to_song(midi, self.beat_div)
        song['events'] = song_to_events(song)
        # song['ids'] = [self.e2i(e) for e in song['events']]
        song['ls_events'] = extract_leadsheet_from_events(
            song['events'],
            song['metadata']['beat_per_bar'],
            self.beat_div
        )
        # song['ls_ids'] = [self.e2i(e) for e in song['ls_events']]

        return song

    def events_to_midi(self, events, beat_per_bar):
        midi = events_to_midi(events, beat_per_bar, self.ticks_per_beat, self.beat_div)
        return midi

    def e2i(self, e):
        ids = [self.vocab.t2i[e[i]] for i in range(Event.WORD_SIZE)]
        return ids

    def i2e(self, ids):
        event = [self.vocab.i2t[ids[i]] for i in range(Event.WORD_SIZE)]
        kwargs = {}
        for t in event:
            if t == self.vocab.i2t[0]:
                continue
            k, v = t.split('_', 1)
            kwargs[k] = v
        event = Event(**kwargs)
        return event

    def get_family_mask(self):
        mask = torch.ones(len(self.vocab), Event.WORD_SIZE).long()
        mask[self.vocab.fmap['spec']] =     torch.LongTensor([1, 1, 0, 0, 0, 0, 0, 0, 0])
        mask[self.vocab.fmap['bar']] =      torch.LongTensor([1, 0, 1, 0, 0, 0, 0, 0, 0])
        mask[self.vocab.fmap['metric']] =   torch.LongTensor([1, 0, 0, 1, 1, 1, 0, 0, 0])
        mask[self.vocab.fmap['note']] =     torch.LongTensor([1, 0, 0, 0, 0, 0, 1, 1, 1])
        return mask

    def get_bar_ranges(self, events):
        bar_idx_list = []
        for i, event in enumerate(events):
            if event.family == 'bar':
                bar_idx_list.append(i)
        bar_idx_list = bar_idx_list + [len(events)]
        bar_idx_list[0] = 0 # the first bar starts at 0
        bar_ranges = []
        for i in range(len(bar_idx_list)-1):
            bar_ranges.append((bar_idx_list[i], bar_idx_list[i+1]))
        return bar_ranges

class Vocab:
    def __init__(self, vacab_file):
        self.vocab = json.loads(vacab_file.read_text())
        self.t2i = {}
        self.i2t = []
        self.fmap = {}

        idx = 0
        self.t2i[self.vocab[0]] = idx
        self.i2t.append(self.vocab[0])
        idx += 1
        for tag, group in self.vocab[1].items():
            for t in group:
                self.t2i[t] = idx
                if tag == 'family':
                    self.fmap[t.split('_', 1)[-1]] = idx
                idx += 1
                self.i2t.append(t)

    def len(self):
        return len(self.i2t)

    def __len__(self):
        return len(self.i2t)

    def __repr__(self):
        out = []
        out.append(f"shared: {self.vocab[0]}")
        for t, words in self.vocab[1].items():
            out.append(f"type: {t}")
            for word in words:
                out.append(f"\t{word}")
        return '\n'.join(out)

class MIDIVocab:
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
        return '\n'.join(self.i2t)

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
        if marker.text.split('_')[0] != 'global' and \
        'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    global_bpm = tempos[0].tempo
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
            marker.text.split('_')[1] == 'bpm':
            global_bpm = int(marker.text.split('_')[2])

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
            duration = max(duration, 1) # dur >= 1

            # append
            note_grid[quant_time].append({
                'note': note,
                'pitch': note.pitch,
                'duration': duration,
                'velocity': note.velocity,
            })

        # sort
        for time in note_grid.keys():
            note_grid[time].sort(key=lambda x: -x['pitch'])

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
        tempo_grid[quant_time] = [tempo] # NOTE: only one tempo per time

    all_bpm = [tempo[0].tempo for _, tempo in tempo_grid.items()]
    assert len(all_bpm) > 0, ' No tempo changes in midi file.'
    average_bpm = sum(all_bpm) / len(all_bpm)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        quant_time = round(label.time / grid_resol)
        label_grid[quant_time] = [label]

    # collect
    song_data = {
        'notes': intsr_gird,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': global_bpm,
            'average_bpm': average_bpm,
            'beat_div': beat_div,
            'beat_per_bar': midi_obj.time_signature_changes[0].numerator,
            # 'last_bar': last_bar,
        }
    }
    return song_data

def song_to_events(song):
    beat_div = song['metadata']['beat_div']
    beat_per_bar = song['metadata']['beat_per_bar']
    grid_per_bar = beat_div * beat_per_bar

    events = [Event(family='spec', spec='ss')]

    max_grid = list(chain(song['tempos'].keys(), song['chords'].keys()))
    for _, v in song['notes'].items():
        max_grid.extend(v.keys())
    max_grid = max(max_grid)

    for bar_i in range(0, max_grid + 1, grid_per_bar):
        events.append(Event(family='bar', bar='tgt'))
        for i in range(bar_i, min(bar_i + grid_per_bar, max_grid + 1)):
            word = Event(family='metric', position=str(i-bar_i), tempo='cont', chord='cont')
            empty = True
            if i in song['chords']:
                chord_items = song['chords'][i][0].text.split('_')
                chord = f'{chord_items[0]}_{chord_items[1]}'
                word.chord = chord
                empty = False
            if i in song['tempos']:
                word.tempo = DEFAULT_BPM_BINS[
                    np.argmin(abs(DEFAULT_BPM_BINS-song['tempos'][i][0].tempo))]
                empty = False
            notes = []
            for _, instr in song['notes'].items():
                if i in instr:
                    for note in instr[i]:
                        duration = DEFAULT_DURATION_RANGE[
                            np.argmin(abs(DEFAULT_DURATION_RANGE-note['duration']))]
                        velocity = DEFAULT_VELOCITY_BINS[
                            np.argmin(abs(DEFAULT_VELOCITY_BINS-note['velocity']))]
                        notes.append(Event(
                            family='note',
                            pitch=note['pitch'],
                            duration=duration,
                            velocity=velocity,
                        ))
                        empty = False

            if not empty:
                events.append(word)
            events.extend(notes)

    events.append(Event(family='spec', spec='se'))

    return events

def events_to_midi(events, beat_per_bar, ticks_per_beat, grid_div):
    assert ticks_per_beat % grid_div == 0
    ticks_per_subbeat = ticks_per_beat // grid_div
    ticks_per_bar = ticks_per_beat * beat_per_bar

    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = ticks_per_beat
    track = miditoolkit.midi.containers.Instrument(program=0, is_drum=False, name='piano')
    midi.instruments = [track]

    bar_tick = 0
    subbeat_tick = 0

    pitch, velocity, duration = 0, 0, 0

    for event in events:
        if event.family == 'spec':
            pass
        elif event.family == 'bar':
            bar_tick += ticks_per_bar
        elif event.family == 'metric':
            subbeat_tick = event.position * ticks_per_subbeat
            tempo = event.tempo
            if tempo != 'ign' and tempo != 'cont':
                m = miditoolkit.midi.containers.TempoChange(time=bar_tick+subbeat_tick, tempo=tempo)
                midi.tempo_changes.append(m)
        elif event.family == 'note':
            pitch = event.pitch
            velocity = event.velocity
            duration = event.duration * ticks_per_subbeat
            n = miditoolkit.Note(
                start=bar_tick+subbeat_tick,
                end=bar_tick+subbeat_tick+duration,
                pitch=pitch,
                velocity=velocity
            )
            midi.instruments[0].notes.append(n)
        else:
            raise ValueError(f'Unknown event: {type(event)}')

    return midi

def extract_leadsheet_from_events(events, beat_per_bar, beat_div, cover_beat=0, min_pitch=60, no_chord=False):
    # algorithm:
    # - skyline
    # - filter out notes with pitch < 60

    grids = []
    for event in events:
        if event.family == 'bar':
            for _ in range(beat_per_bar * beat_div):
                grids.append(list())

    grid_idx = 0
    bar_count = 0
    subbeat = 0
    for event in events:
        if event.family == 'spec':
            grids[grid_idx].append(event)
        elif event.family == 'bar':
            grid_idx = bar_count * (beat_per_bar * beat_div)
            bar_count += 1
            subbeat = 0

            event = deepcopy(event)
            event.bar = 'src'
            grids[grid_idx].append(event)
        elif event.family == 'metric':
            if no_chord:
                event = deepcopy(event)
                event.chord = 'cont'
            subbeat = event.position
            grids[grid_idx + subbeat].append(event)
        elif event.family == 'note':
            if event.pitch >= min_pitch: # only keep notes with pitch >= min_pitch
                no_vel = deepcopy(event)
                no_vel.velocity = LS_DEFAULT_VELOCITY
                grids[grid_idx + subbeat].append(no_vel)

    # select the highest note
    for i, grid in enumerate(grids):
        notes = [e for e in grid if e.family == 'note']
        notes.sort(key=lambda x: -x.pitch)
        grids[i] = [e for e in grid if not e.family == 'note']
        if len(notes) > 0:
            grids[i].append(notes[0])

    # remove covered notes
    covered_by = [-1] * len(grids)
    for i, grid in enumerate(grids):
        if len(grid) == 0 or not grid[-1].family == 'note':
            continue
        note = grid[-1]
        if i + note.duration > len(covered_by):
            covered_by += [-1] * (i + note.duration - len(covered_by))
        # we assume a note which span more than N beats can not cover other notes
        for j in range(i, i+min(note.duration, beat_div*cover_beat)):
            c = covered_by[j]
            if c == -1 or note.pitch >= grids[c][-1].pitch:
                covered_by[j] = i

    for i, grid in enumerate(grids):
        if len(grid) == 0 or not grid[-1].family == 'note':
            continue
        note = grid[-1]
        covered = True
        for j in range(i, i+note.duration):
            if covered_by[j] in [-1, i]:
                covered = False
        if covered:
            grids[i].pop()

    # remove useless metric
    for i, grid in enumerate(grids):
        if len(grid) == 0 or not grid[-1].family == 'metric':
            continue
        metric = grid[-1]
        assert metric.family == 'metric'
        if metric.chord == 'cont' and metric.tempo == 'cont':
            grids[i].pop()
            assert len(grid) == 0 or grid[-1].family == 'bar'

    return list(chain(*grids))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cmd_gen_vocab = subparsers.add_parser('gen_vocab')
    cmd_gen_vocab.add_argument('--output_file', type=Path, required=True)
    ca = parser.parse_args()

    if ca.command is None:
        parser.print_help()
        exit()
    elif ca.command == 'gen_vocab':
        vocab = gen_vocab()
        ca.output_file.write_text(json.dumps(vocab, indent=2))
        vocab = Vocab(ca.output_file)
        print(vocab)
        print("vocab size:", vocab.len())

        midi_vocab = gen_vocab_midi()
        midi_vocab_file = ca.output_file.parent / (ca.output_file.stem + '_midi.json')
        midi_vocab_file.write_text(json.dumps(midi_vocab, indent=2))
    else:
        raise ValueError(f'Unknown command: {ca.command}')

