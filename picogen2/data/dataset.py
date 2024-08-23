import functools
import json
import tempfile
from random import randrange

import miditoolkit
import numpy as np
import pretty_midi
import torch
import yaml
from chorder import Dechorder
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .. import utils
from ..model import PiCoGenDecoder
from ..repr import Event

PITCH_MAP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


class Pop2PianoDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        processed_dir,
        tokenizer,
    ):
        self.dataset_dir = dataset_dir
        self.processed_dir = processed_dir
        self.tokenizer = tokenizer
        self.song_list = sorted(list(map(lambda d: d.name, self.processed_dir.iterdir())))

        print("Generating piano events ...")
        self.build_data()

    def __len__(self):
        return len(self.song_list)

    def __getitem__(self, idx):
        return self._get_song(self.song_list[idx])

    def _get_song(self, idx):
        dataset_dir = self.dataset_dir / str(idx).zfill(4)
        processed_dir = self.processed_dir / str(idx).zfill(4)
        song = {}
        song["source"] = processed_dir / "piano.mid"
        song["metadata"] = yaml.safe_load((dataset_dir / "info.yaml").read_text())
        song["piano_events"] = utils.pickle_load(
            processed_dir / f"piano_events_{self.tokenizer.beat_div}.pkl"
        )
        song["piano_sheetsage"] = np.load(processed_dir / "piano_sheetsage.npz")
        song["piano_beat"] = json.loads((processed_dir / "piano_beat.json").read_text())
        song["pop_sheetsage"] = np.load(processed_dir / "song_sheetsage.npz")
        song["pop_beat"] = json.loads((processed_dir / "song_beat.json").read_text())
        song["time_map"] = json.loads((processed_dir / "align_info.json").read_text())["time_map"]

        return song

    def build_data(self):
        candidates = []
        for song_idx in tqdm(self.song_list):
            piano_events_file = (
                self.processed_dir / song_idx / f"piano_events_{self.tokenizer.beat_div}.pkl"
            )
            if not piano_events_file.exists():
                candidates.append(song_idx)

        def process_song(song_idx, processed_dir, tokenizer):
            piano_events_file = processed_dir / song_idx / f"piano_events_{tokenizer.beat_div}.pkl"
            piano_midi_file = processed_dir / song_idx / "piano.mid"
            piano_beat_info = json.loads((processed_dir / song_idx / "piano_beat.json").read_text())
            piano_events = self._get_song_events(piano_midi_file, piano_beat_info, tokenizer)
            utils.pickle_save(piano_events, piano_events_file)

        if len(candidates) > 0:
            Parallel(n_jobs=-1)(
                delayed(process_song)(song_idx, self.processed_dir, self.tokenizer)
                for song_idx in tqdm(candidates)
            )

    @classmethod
    def _get_song_events(cls, midi_file, beat_info, tokenizer):
        midi_objs = pretty_midi.PrettyMIDI(str(midi_file))
        beats = np.array(beat_info["beats"])
        downbeats_idx = utils.get_downbeat_indices(beats, beat_info["downbeats"])

        # each beat containes 4 subbeats
        beat_div = tokenizer.beat_div
        subbeats = [
            beats[i] + (beats[i + 1] - beats[i]) * j / beat_div
            for i in range(len(beats) - 1)
            for j in range(beat_div)
        ]
        subbeats.extend(
            [beats[-1] + (beats[-1] - beats[-2]) * j / beat_div for j in range(beat_div)]
        )

        note_grid = [list() for _ in range(len(subbeats))]
        tempo_grid = [-1] * len(subbeats)
        chord_grid = [None] * len(subbeats)

        for i, inst in enumerate(midi_objs.instruments):
            if inst.is_drum:
                continue
            for note in inst.notes:
                # assert note.start >= beats_times[0] and note.end <= beats_times[-1]
                if note.start < subbeats[0]:
                    continue
                onset = np.argmin(np.abs(subbeats - note.start))
                offset = np.argmin(np.abs(subbeats - note.end))
                duration = offset - onset
                pitch, velocity = note.pitch, note.velocity
                # print(f'{i}: {onset} - {offset}, {duration}, {pitch}, {velocity}')
                note_grid[onset].append(dict(pitch=pitch, duration=duration, velocity=velocity))

        for i in range(len(note_grid)):
            note_grid[i] = sorted(note_grid[i], key=lambda x: -x["pitch"])

        tempos = np.round(60 / np.diff(beats))
        tempo_grid[0] = tempos[0]
        for i in range(1, len(tempos)):
            if tempos[i] != tempos[i - 1]:
                tempo_grid[i * beat_div] = tempos[i]

        chords = cls._get_chords(note_grid, beat_div)

        prev_chord = None
        for i in range(len(chords)):
            if chords[i] is None or not chords[i].is_complete():
                chord = Event(etype="metric", value="chord_N_N")
            else:
                chord_text = PITCH_MAP[chords[i].root_pc] + "_" + chords[i].quality
                chord = Event(etype="metric", value=f"chord_{chord_text}")

            if chord != prev_chord:
                chord_grid[i * beat_div] = chord
                prev_chord = chord

        events = []
        # TODO: handle upbeat
        for i in range(len(downbeats_idx) - 1):
            bar_start, bar_end = downbeats_idx[i], downbeats_idx[i + 1]
            events.append(Event(etype="bar", value="bar_start"))
            bar_len = bar_end - bar_start
            if 1 <= bar_len <= 4:
                bar_text = f"bar_{bar_len}"
            else:
                bar_text = "bar_N"
            events.append(Event(etype="bar", value=bar_text))

            bar_start = bar_start * beat_div
            bar_end = bar_end * beat_div

            for pos in range(bar_start, bar_end):
                tmp = []
                if chord_grid[pos] is not None:
                    tmp.append(chord_grid[pos])
                if tempo_grid[pos] != -1:
                    tmp.append(tokenizer.get_tempo_event(tempo_grid[pos]))
                if len(note_grid[pos]) > 0:
                    for note in note_grid[pos]:
                        pitch, duration, velocity = (
                            note["pitch"],
                            note["duration"],
                            note["velocity"],
                        )
                        tmp.append(Event(etype="note", value=f"pitch_{pitch}"))
                        tmp.append(tokenizer.get_duration_event(duration))
                        tmp.append(tokenizer.get_velocity_event(velocity))
                if len(tmp) > 0:
                    events.append(Event(etype="metric", value=f"position_{pos - bar_start}"))
                    events.extend(tmp)
            events.append(Event(etype="bar", value="bar_end"))

        header = [Event(etype="spec", value="spec_ss")]
        header.append(tokenizer.get_tempo_event(tempos[0]))
        events = header + events

        events.append(Event(etype="spec", value="spec_se"))
        return events

    @classmethod
    def _get_chords(cls, note_grid, beat_div):
        with tempfile.NamedTemporaryFile() as tmpf:
            mido_obj = miditoolkit.MidiFile()
            beat_resol = mido_obj.ticks_per_beat
            subbeat_resol = beat_resol // beat_div

            track = miditoolkit.Instrument(program=0, is_drum=False)
            mido_obj.instruments = [track]

            for i in range(len(note_grid)):
                cur_tick = int(subbeat_resol * i)
                for note in note_grid[i]:
                    pitch, duration, velocity = (
                        note["pitch"],
                        note["duration"],
                        note["velocity"],
                    )
                    track.notes.append(
                        miditoolkit.Note(
                            start=cur_tick,
                            end=cur_tick + duration * subbeat_resol,
                            pitch=pitch,
                            velocity=velocity,
                        )
                    )

            mido_obj.dump(tmpf.name)
            mido_obj = miditoolkit.MidiFile(tmpf.name)

            chords = Dechorder.dechord(mido_obj)
            return chords

    @classmethod
    def _get_stats(cls, songs):
        len_events = [len(song["events"]) for song in songs]
        len_events_bar = []
        for song in songs:
            bars = [[]]
            for eid in song["events"]:
                if eid.etype == "bar":
                    bars.append([])
                bars[-1].append(eid)
            bars = bars[1:]
            for bar in bars:
                if len(bar) > 0:
                    len_events_bar.append(len(bar))

        stats = {}
        stats["num_songs"] = len(songs)
        stats["mean_events"] = np.mean(len_events)
        stats["std_events"] = np.std(len_events)
        stats["mean_events_bar"] = np.mean(len_events_bar)
        stats["std_events_bar"] = np.std(len_events_bar)

        return stats

    @classmethod
    def get_dataloader(cls, tokenizer, max_seq_len, *args, fn_name="mix", **kwargs):
        fn_map = {
            "piano": cls.collate_fn_piano,
            "pair": cls.collate_fn_pair,
            "mix": cls.collate_fn_mix,
        }
        return DataLoader(
            *args,
            **kwargs,
            collate_fn=functools.partial(
                cls.collate_fn_mix,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            ),
        )

    @staticmethod
    def collate_fn_piano(batch, tokenizer, max_seq_len):
        # random pick a segment
        # raondom mask
        # transpose

        input_segs = []
        label_segs = []
        cls_id_segs = []
        need_encode_segs = []

        for song in batch:
            ids = [tokenizer.e2i(e) for e in song["piano_events"]]
            bar_ranges = tokenizer.get_bar_ranges(song["piano_events"], from_start=False)
            # downbeats = song["piano_beat_info"]["downbeats"]
            downbeats_idx = utils.get_downbeat_indices(
                song["piano_beat"]["beats"], song["piano_beat"]["downbeats"]
            )
            sheetsage_melody = np.array(song["piano_sheetsage"]["melody"])
            sheetsage_harmony = np.array(song["piano_sheetsage"]["harmony"])

            assert len(downbeats_idx) == len(bar_ranges) + 1

            input_seg = []
            cls_ids = []
            need_encode = []
            ls_bar_idx = []
            label_seg = []

            input_seg.extend(ids[: bar_ranges[0][0]])  # NOTE: meta events
            cls_ids.extend([PiCoGenDecoder.InputClass.TARGET.value] * len(input_seg))
            need_encode.extend([0] * len(input_seg))
            label_seg.extend([-100] * len(input_seg))

            for i in range(len(bar_ranges)):
                downbeat_start, downbeat_end = downbeats_idx[i], downbeats_idx[i + 1]

                ls_bar_idx.append(len(input_seg))
                for j in range(
                    downbeat_start * tokenizer.beat_div,
                    downbeat_end * tokenizer.beat_div,
                ):
                    input_seg.append(
                        (
                            sheetsage_melody[j],
                            sheetsage_harmony[j],
                        )
                    )
                    cls_ids.append(PiCoGenDecoder.InputClass.CONDITION.value)
                    need_encode.append(1)
                    # cond_mask.append(1)
                    label_seg.append(-100)
                if i == len(bar_ranges) - 1:  # NOTE: add song_end to the last bar condition
                    input_seg.append(tokenizer.e2i(Event(etype="spec", value="spec_se")))
                    cls_ids.append(PiCoGenDecoder.InputClass.CONDITION.value)
                    need_encode.append(0)
                    label_seg.append(-100)

                tgt_bar_start, tgt_bar_end = bar_ranges[i]
                ids_slice = ids[tgt_bar_start:tgt_bar_end]
                input_seg.extend(ids_slice)
                cls_ids.extend([PiCoGenDecoder.InputClass.TARGET.value] * len(ids_slice))
                need_encode.extend([0] * len(ids_slice))

                label = ids_slice.copy()
                label[0] = -100  # NOTE: bar start
                label_seg.extend(label)

            assert len(input_seg) == len(cls_ids)

            if len(input_seg) // (max_seq_len // 2) == 0:
                idx_from = 0
            else:
                idx_from = randrange(0, len(input_seg) // (max_seq_len // 2)) * (max_seq_len // 2)

            # find previous bar_src
            for i in range(idx_from):
                if idx_from in ls_bar_idx:
                    break
                idx_from -= 1

            idx_to = min(idx_from + max_seq_len, len(input_seg))
            input_seg = [tokenizer.e2i(Event(etype="spec", value="spec_bos"))] + input_seg[
                idx_from:idx_to
            ]
            cls_id_seg = [0] + cls_ids[idx_from:idx_to]
            need_encode = [0] + need_encode[idx_from:idx_to]
            label_seg = [-100] + label_seg[idx_from:idx_to]  # +1 for bos
            assert len(input_seg) == len(label_seg)

            # randomly mask
            pass

            # transpose
            pass

            input_seg = input_seg + [0] * (max_seq_len + 1 - len(input_seg))  # +1 for bos
            cls_id_seg = cls_id_seg + [0] * (max_seq_len + 1 - len(cls_id_seg))  # +1 for bos
            need_encode = need_encode + [0] * (max_seq_len + 1 - len(need_encode))  # +1 for bos
            label_seg = label_seg + [-100] * (max_seq_len + 1 - len(label_seg))  # +1 for bos

            input_seg = input_seg[:-1]
            cls_id_seg = cls_id_seg[:-1]
            need_encode = need_encode[:-1]
            label_seg = label_seg[1:]
            assert len(input_seg) == max_seq_len
            assert len(input_seg) == len(label_seg) == len(cls_id_seg) == len(need_encode)

            # append
            input_segs.append(input_seg)
            label_segs.append(torch.LongTensor(label_seg))
            cls_id_segs.append(torch.LongTensor(cls_id_seg))
            need_encode_segs.append(torch.BoolTensor(need_encode))

        label_segs = torch.stack(label_segs, dim=0)
        cls_id_segs = torch.stack(cls_id_segs, dim=0)
        need_encode_segs = torch.stack(need_encode_segs, dim=0)

        B, L = cls_id_segs.shape
        input_ids = torch.zeros(B, L).long()
        input_cond_embs = torch.zeros(B, L, 2, 512).float()
        for b in range(B):
            for ll in range(L):
                if need_encode_segs[b, ll]:
                    emb = torch.FloatTensor(np.array(input_segs[b][ll]))
                    input_cond_embs[b, ll] = emb
                else:
                    input_ids[b, ll] = input_segs[b][ll]

        return {
            "source": [song["source"] for song in batch],
            "metadata": [song["metadata"] for song in batch],
            "input_segs": input_segs,
            "input_ids": input_ids,
            "input_cond_embs": input_cond_embs,
            "label_ids": label_segs,
            "input_cls_ids": cls_id_segs,
            "need_encode": need_encode_segs,
        }

    @staticmethod
    def collate_fn_pair(batch, tokenizer, max_seq_len):
        # random pick a segment
        # raondom mask
        # transpose

        input_segs = []
        label_segs = []
        cls_id_segs = []
        need_encode_segs = []

        for song in batch:
            ids = [tokenizer.e2i(e) for e in song["piano_events"]]
            bar_ranges = tokenizer.get_bar_ranges(song["piano_events"], from_start=False)
            # downbeats_idx = song["piano_beat_info"]["downbeats"]
            downbeats_idx = utils.get_downbeat_indices(
                song["piano_beat"]["beats"], song["piano_beat"]["downbeats"]
            )
            sheetsage_melody = np.array(song["pop_sheetsage"]["melody"])
            sheetsage_harmony = np.array(song["pop_sheetsage"]["harmony"])
            time_map = song["time_map"]

            assert len(downbeats_idx) == len(bar_ranges) + 1
            new_times, original_times = time_map["piano"], time_map["song"]
            piano_beats_times = song["piano_beat"]["beats"].copy()
            pop_beats_times = np.array(song["pop_beat"]["beats"])
            adjusted_piano_beats_times = np.interp(piano_beats_times, original_times, new_times)

            beat_map = []
            for adjusted_beat_time in adjusted_piano_beats_times:
                idx = np.argmin(np.abs(pop_beats_times - adjusted_beat_time))
                beat_map.append(idx)

            mapped_downbeats = [beat_map[b] for b in downbeats_idx]
            mapped_bar_len = np.diff(mapped_downbeats)

            input_seg = []
            cls_ids = []
            need_encode = []
            # cond_mask = []
            ls_bar_idx = []
            label_seg = []

            input_seg.extend(ids[: bar_ranges[0][0]])  # NOTE: meta events
            cls_ids.extend([PiCoGenDecoder.InputClass.TARGET.value] * len(input_seg))
            need_encode.extend([0] * len(input_seg))
            label_seg.extend([-100] * len(input_seg))

            for i in range(len(bar_ranges)):
                if mapped_bar_len[i] == 0:
                    break

                downbeat_start, downbeat_end = (
                    mapped_downbeats[i],
                    mapped_downbeats[i + 1],
                )
                ls_bar_idx.append(len(input_seg))
                for j in range(
                    downbeat_start * tokenizer.beat_div,
                    downbeat_end * tokenizer.beat_div,
                ):
                    input_seg.append((sheetsage_melody[j], sheetsage_harmony[j]))
                    cls_ids.append(PiCoGenDecoder.InputClass.CONDITION.value)
                    need_encode.append(1)
                    label_seg.append(-100)
                if i == len(bar_ranges) - 1:  # NOTE: add song_end to the last bar condition
                    input_seg.append(tokenizer.e2i(Event(etype="spec", value="spec_se")))
                    cls_ids.append(PiCoGenDecoder.InputClass.CONDITION.value)
                    need_encode.append(0)
                    label_seg.append(-100)

                tgt_bar_start, tgt_bar_end = bar_ranges[i]
                ids_slice = ids[tgt_bar_start:tgt_bar_end]
                input_seg.extend(ids_slice)
                cls_ids.extend([PiCoGenDecoder.InputClass.TARGET.value] * len(ids_slice))
                need_encode.extend([0] * len(ids_slice))

                label = ids_slice.copy()
                label[0] = -100  # NOTE: bar start
                label_seg.extend(label)

            assert len(input_seg) == len(cls_ids)

            if len(input_seg) // (max_seq_len // 2) == 0:
                idx_from = 0
            else:
                idx_from = randrange(0, len(input_seg) // (max_seq_len // 2)) * (max_seq_len // 2)

            # find previous bar_src
            for i in range(idx_from):
                if idx_from in ls_bar_idx:
                    break
                idx_from -= 1

            idx_to = min(idx_from + max_seq_len, len(input_seg))
            input_seg = [tokenizer.e2i(Event(etype="spec", value="spec_bos"))] + input_seg[
                idx_from:idx_to
            ]
            cls_id_seg = [0] + cls_ids[idx_from:idx_to]
            need_encode = [0] + need_encode[idx_from:idx_to]
            label_seg = [-100] + label_seg[idx_from:idx_to]  # +1 for bos
            assert len(input_seg) == len(label_seg)

            # randomly mask
            pass

            # transpose
            pass

            input_seg = input_seg + [0] * (max_seq_len + 1 - len(input_seg))  # +1 for bos
            cls_id_seg = cls_id_seg + [0] * (max_seq_len + 1 - len(cls_id_seg))  # +1 for bos
            need_encode = need_encode + [0] * (max_seq_len + 1 - len(need_encode))  # +1 for bos
            label_seg = label_seg + [-100] * (max_seq_len + 1 - len(label_seg))  # +1 for bos

            input_seg = input_seg[:-1]
            cls_id_seg = cls_id_seg[:-1]
            need_encode = need_encode[:-1]
            label_seg = label_seg[1:]
            assert len(input_seg) == max_seq_len
            assert len(input_seg) == len(label_seg) == len(cls_id_seg) == len(need_encode)

            # append
            input_segs.append(input_seg)
            label_segs.append(torch.LongTensor(label_seg))
            cls_id_segs.append(torch.LongTensor(cls_id_seg))
            need_encode_segs.append(torch.BoolTensor(need_encode))

        label_segs = torch.stack(label_segs, dim=0)
        cls_id_segs = torch.stack(cls_id_segs, dim=0)
        need_encode_segs = torch.stack(need_encode_segs, dim=0)

        B, L = cls_id_segs.shape
        input_ids = torch.zeros(B, L).long()
        input_cond_embs = torch.zeros(B, L, 2, 512).float()
        for b in range(B):
            for ll in range(L):
                if need_encode_segs[b, ll]:
                    emb = torch.FloatTensor(np.array(input_segs[b][ll]))
                    input_cond_embs[b, ll] = emb
                else:
                    input_ids[b, ll] = input_segs[b][ll]

        return {
            "source": [song["source"] for song in batch],
            "metadata": [song["metadata"] for song in batch],
            "input_segs": input_segs,
            "input_ids": input_ids,
            "input_cond_embs": input_cond_embs,
            "label_ids": label_segs,
            "input_cls_ids": cls_id_segs,
            "need_encode": need_encode_segs,
        }

    @staticmethod
    def collate_fn_mix(batch, tokenizer, max_seq_len):
        piano_batch = Pop2PianoDataset.collate_fn_piano(batch, tokenizer, max_seq_len)
        pair_batch = Pop2PianoDataset.collate_fn_pair(batch, tokenizer, max_seq_len)
        return {
            "piano": piano_batch,
            "pair": pair_batch,
        }
