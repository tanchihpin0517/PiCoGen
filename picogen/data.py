from copy import deepcopy
from random import randrange, randint
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import functools
import numpy as np
import argparse
import miditoolkit
import json

from .vocab import Vocab
from .repr import Event, Tokenizer
from . import utils

class AILabs1k7Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        use_cache=True,
    ):
        self.songs = self.load_data(
            data_dir,
            tokenizer,
            use_cache=use_cache,
        )

        len_events = [len(song['events']) for song in self.songs]
        ls_events = [len(song['ls_events']) for song in self.songs]

        print('[mean and std]')
        print(f'\torig music data: {np.mean(len_events)}, {np.std(len_events)}')
        print(f'\torig lead sheet data: {np.mean(ls_events)}, {np.std(ls_events)}')

    def load_data(
        self,
        data_dir,
        tokenizer,
        use_cache=True,
    ):
        cache_file = data_dir.parent / f'{data_dir.stem}_cache_{tokenizer.beat_div}.pkl'
        if use_cache:
            if cache_file.exists():
                print(f'loading cached data from {cache_file}')
                return utils.pickle_load(cache_file)

        midi_files = list((data_dir / "midi_analyzed").glob('**/*.mid'))
        midi_files.sort(key = lambda x: int(Path(x).stem.split('_')[0]))

        map_args = list(zip(midi_files, [tokenizer]*len(midi_files)))
        cache_out_dir = cache_file.parent / cache_file.stem
        cache_out_dir.mkdir(exist_ok=True)
        songs = []

        with mp.Pool() as pool:
            for i, song in enumerate(tqdm(pool.imap(self.load_data_map, map_args), total=len(map_args), dynamic_ncols=True)):
                songs.append(song)

                song_dir = cache_out_dir / f'{i}'
                song_dir.mkdir(exist_ok=True)
                shutil.copy(song['source'], song_dir / 'source.mid')
                (song_dir / 'meta.txt').write_text(str(song['metadata']))

                (song_dir / 'events.txt').write_text("\n".join(map(str, song['events'])))
                (song_dir / 'events_raw.txt').write_text(
                    "\n".join(map(lambda e: str(e.data), song['events'])))
                # (song_dir / 'ids.txt').write_text("\n".join(map(str, song['ids'])))
                tokenizer.events_to_midi(song['events'], song['metadata']['beat_per_bar']).dump(song_dir / 'events.mid')

                (song_dir / 'ls_events.txt').write_text("\n".join(map(str, song['ls_events'])))
                (song_dir / 'ls_events_raw.txt').write_text(
                    "\n".join(map(lambda e: str(e.data), song['ls_events'])))
                # (song_dir / 'ls_ids.txt').write_text("\n".join(map(str, song['ls_ids'])))
                tokenizer.events_to_midi(song['ls_events'], song['metadata']['beat_per_bar']).dump(song_dir / 'ls_events.mid')

        utils.pickle_save(songs, cache_file)

        return songs

    @staticmethod
    def load_data_map(args):
        midi_file, tokenizer = args

        midi_objs = miditoolkit.MidiFile(midi_file)
        song = tokenizer.get_song_from_midi(midi_objs)

        song['source'] = midi_file

        return song

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        return self.songs[idx]

    @classmethod
    def get_dataloader(cls, tokenizer, max_seq_len, *args, **kwargs):
        return DataLoader(
            *args,
            **kwargs,
            collate_fn=functools.partial(
                cls.collate_fn,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            ),
        )

    @staticmethod
    def collate_fn(batch, tokenizer, max_seq_len):
        tgt_segs = []
        label_segs = []

        for song in batch:
            ids = [tokenizer.e2i(e) for e in song['events']]
            bar_ranges = tokenizer.get_bar_ranges(song['events'])
            ls_ids = [tokenizer.e2i(e) for e in song['ls_events']]
            ls_bar_ranges = tokenizer.get_bar_ranges(song['ls_events'])
            assert len(bar_ranges) == len(ls_bar_ranges)

            # bos bar_src ...(src)... bar_tgt ...(tgt)... bar_src ...
            mixed_ids = []
            cond_mask = []
            for i in range(len(bar_ranges)):
                start, end = bar_ranges[i]
                ls_start, ls_end = ls_bar_ranges[i]

                mixed_ids.extend(ls_ids[ls_start:ls_end])
                mask = [1] * (ls_end - ls_start)
                mask[0] = 0
                cond_mask.extend(mask)

                mixed_ids.extend(ids[start:end])
                mask = [0] * (end - start)
                mask[0] = 1
                cond_mask.extend(mask)
            assert len(mixed_ids) == len(cond_mask)


            if len(mixed_ids) // (max_seq_len // 2) == 0:
                idx_from = 0
            else:
                idx_from = randrange(0, len(mixed_ids) // (max_seq_len // 2)) * (max_seq_len // 2)

            # find previous bar_src
            bar_src_id = tokenizer.e2i(Event(family='bar', bar='src'))
            for i in range(idx_from):
                if mixed_ids[idx_from] == bar_src_id:
                    break
                idx_from -= 1

            idx_to = min(idx_from + max_seq_len, len(mixed_ids))
            tgt_seg = mixed_ids[idx_from:idx_to]
            tgt_seg = [tokenizer.e2i(Event(family='spec', spec='bos'))] + tgt_seg
            cond_mask = cond_mask[idx_from:idx_to]


            # randomly mask
            pass

            # transpose
            pass

            # to tensor
            tgt_seg = torch.LongTensor(tgt_seg)
            cond_mask = torch.BoolTensor(cond_mask)[:, None].expand(-1, tgt_seg.shape[-1])

            # pad
            pad_id = -1
            tgt_seg = F.pad(tgt_seg, (0, 0, 0, max_seq_len+1-len(tgt_seg)), 'constant', pad_id) # +1 for bos
            cond_mask = F.pad(cond_mask, (0, 0, 0, max_seq_len-cond_mask.shape[-2]), 'constant', False)
            label_seg = tgt_seg.detach().clone()[1:, :]
            tgt_seg = tgt_seg[:-1, :]
            assert len(tgt_seg) == max_seq_len

            # label ignore
            ign_id = tokenizer.vocab.t2i['ign']
            tgt_seg[tgt_seg == pad_id] = 0
            label_seg[label_seg == pad_id] = -100
            label_seg[label_seg == ign_id] = -100
            label_seg[cond_mask] = -100

            # append
            tgt_segs.append(tgt_seg)
            label_segs.append(label_seg)

        tgt_segs = torch.stack(tgt_segs, dim=0)
        label_segs = torch.stack(label_segs, dim=0)

        return {
            'source': [song['source'] for song in batch],
            'metadata': [song['metadata'] for song in batch],
            'tgt_ids': tgt_segs,
            'label_ids': label_segs,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cmd_cache = subparsers.add_parser('gen_cache')
    cmd_cache.add_argument('--data_dir', type=Path, required=True)
    cmd_cache.add_argument('--vocab_file', type=Path, required=True)
    cmd_cache.add_argument('--config', type=Path, required=True)
    ca = parser.parse_args()

    if ca.command is None:
        parser.print_help()
        exit()
    elif ca.command == 'gen_cache':
        config = json.loads(ca.config.read_text())
        dataset = AILabs1k7Dataset(
            ca.data_dir,
            Tokenizer(ca.vocab_file, config['beat_div'], config['ticks_per_beat']),
            use_cache=False,
        )
    else:
        raise ValueError(f'unknown command {ca.command}')
