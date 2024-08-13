import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from mirtoolkit import beat_this, sheetsage
from tqdm import tqdm

from .data.download import ytdlp_download
from .model import PiCoGenDecoder
from .repr import Event
from .utils import downbeat_time_to_index


def download(input_url: str, output_file: Path):
    tmp_dir = tempfile.TemporaryDirectory()
    audio_file = Path(tmp_dir.name) / "song.mp3"
    ytdlp_download(input_url, audio_file)
    shutil.copy(audio_file, output_file)


@torch.no_grad()
def detect_beat(audio_file: Path, output_file: Path):
    beats, downbeats = beat_this.detect(audio_file)
    beat_info = {"beats": beats.tolist(), "downbeats": downbeats.tolist()}
    output_file.write_text(json.dumps(beat_info, indent=4))


@torch.no_grad()
def extract_sheetsage_feature(audio_file: Path, output_file: Path, beat_file: Path):
    beat_info = json.loads(beat_file.read_text())
    sheetsage_output = sheetsage.infer(audio_path=audio_file, beat_information=beat_info)
    np.savez_compressed(
        output_file,
        melody=sheetsage_output["melody_last_hidden_state"],
        harmony=sheetsage_output["harmony_last_hidden_state"],
    )


@torch.no_grad()
def decode(
    model,
    tokenizer,
    beat_information,
    melody_last_embs,
    harmony_last_embs,
    max_bar_num=None,
    max_token_num=None,
    temperature=1.0,
    device=None,
):
    if device is None:
        device = model.parameters().__next__().device

    starting_bpm = 60 / np.diff(np.array(beat_information["beats"])[:2]).mean()

    TARGET = PiCoGenDecoder.InputClass.TARGET.value
    CONDITION = PiCoGenDecoder.InputClass.CONDITION.value

    out_events = [
        Event(etype="spec", value="spec_ss"),
        tokenizer.get_tempo_event(starting_bpm),
    ]

    input_seg = [tokenizer.e2i(Event(etype="spec", value="spec_bos"))]
    input_seg.extend([tokenizer.e2i(e) for e in out_events])  # song start, global tempo
    need_encode_seg = [0] * len(input_seg)
    input_cls_seg = [TARGET] * len(input_seg)

    end_event = Event(etype="spec", value="spec_se")
    bar_start_event = Event(etype="bar", value="bar_start")
    bar_end_event = Event(etype="bar", value="bar_end")

    total_beats = len(beat_information["beats"])
    downbeats = downbeat_time_to_index(beat_information["beats"], beat_information["downbeats"])
    if downbeats[-1] < total_beats:
        downbeats.append(total_beats - 1)
    if max_bar_num is not None:
        downbeats = downbeats[: max_bar_num + 1]

    pbar = tqdm(total=len(downbeats) - 1)
    for bar_i, b in enumerate(range(len(downbeats) - 1)):
        last_past_kv = None

        # NOTE: upbeat is handled by SheetSage
        downbeat_start, downbeat_end = downbeats[b], downbeats[b + 1]
        # downbeat_start, downbeat_end = downbeats[b]-downbeats[0], downbeats[b+1]-downbeats[0]
        for j in range(downbeat_start * tokenizer.beat_div, downbeat_end * tokenizer.beat_div):
            input_seg.append((melody_last_embs[j], harmony_last_embs[j]))
            input_cls_seg.append(CONDITION)
            need_encode_seg.append(1)
        if b == len(downbeats) - 2:  # NOTE: add song_end to the last bar condition
            input_seg.append(tokenizer.e2i(Event(etype="spec", value="spec_se")))
            input_cls_seg.append(CONDITION)
            need_encode_seg.append(0)

        input_seg.append(tokenizer.e2i(bar_start_event))
        need_encode_seg.append(0)
        input_cls_seg.append(TARGET)
        out_events.append(bar_start_event)

        while True:  # generate one bar
            if len(input_seg) > model.hp.max_seq_len:
                input_seg = input_seg[-model.hp.max_seq_len // 2 :]
                input_seg[0] = tokenizer.e2i(Event(etype="spec", value="spec_bos"))
                need_encode_seg = need_encode_seg[-model.hp.max_seq_len // 2 :]
                need_encode_seg[0] = 0
                input_cls_seg = input_cls_seg[-model.hp.max_seq_len // 2 :]
                input_cls_seg[0] = TARGET
                last_past_kv = None

            input_cls_ids = torch.LongTensor(input_cls_seg)[None, :].to(device)
            need_encode = torch.BoolTensor(need_encode_seg)[None, :].to(device)

            output_ids, past_kv = model.generate(
                input_seg=[input_seg],
                input_cls_ids=input_cls_ids,
                need_encode=need_encode,
                kv_cache=last_past_kv,
                temperature=temperature,
            )
            out_id = output_ids[0][-1].item()
            out_event = tokenizer.i2e(out_id)

            out_events.append(out_event)
            input_seg.append(out_id)
            input_cls_seg.append(TARGET)
            need_encode_seg.append(0)

            if out_event in (bar_end_event, end_event):
                break
            last_past_kv = past_kv

            pbar.set_description(f"length: {len(out_events)}({len(input_seg)})")

        pbar.update(1)

        if max_token_num is not None and len(input_seg) > max_token_num:
            break

    pbar.close()
    return out_events
