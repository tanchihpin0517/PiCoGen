import json
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import questionary
import torch
import torch.nn.functional as F

_logger = None
_level = None


@dataclass
class HyperParam:
    beat_div: int
    ticks_per_beat: int

    seed: int
    learning_rate: float
    learning_rate_min: float
    adam_b1: float
    adam_b2: float
    sched_T: int
    warmup_epochs: int

    vocab_size: int
    token_class: int
    condition_class: int
    d_model: int
    d_bottleneck: int
    num_layers: int
    num_layers_encoder: int
    num_heads: int
    activation: str
    dropout: float
    max_seq_len: int
    max_position_embeddings: int

    loss_weight: float = 1.0


def _get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("picogen2")

    return _logger


class Logger:
    def setLevel(self, level):
        global _level, _logger
        _level = level.upper()
        _get_logger().setLevel(_level)

    def __getattr__(self, name):
        return getattr(_get_logger(), name)

    def __repr__(self):
        return repr(_get_logger())


logger = Logger()


def check_task_done(task: str, output_dir: Path):
    done_file = output_dir / f"done_{task}"
    return done_file.exists()


def mark_task_done(task: str, output_dir: Path):
    done_file = output_dir / f"done_{task}"
    done_file.touch()


def song_dir_name(index: int):
    return "{:04d}".format(index)


def load_config(config_file):
    config = json.loads(config_file.read_text())
    hp = HyperParam(**config)
    logger.info("checkpoint model config:")
    for v in fields(hp):
        logger.info(f"\t{v.name}: {getattr(hp, v.name)}")
    return hp


def load_checkpoint(filepath: Path, device):
    assert filepath.is_file()
    logger.info("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    logger.info("Done.")
    return checkpoint_dict


def query_mkdir(path):
    if not path.exists():
        confirm = questionary.confirm(
            f'Directory "{path}" does not exist, create?', default=False
        ).ask()
        if confirm:
            path.mkdir(parents=True)
            return True
        else:
            print(f'Directory "{path}" does not exist. Exit.')
            return False
    return True


def downbeat_time_to_index(beats, downbeats):
    downbeat_indices = []
    beats = np.array(beats)
    for downbeat in downbeats:
        idx = np.argmin(np.abs(beats - downbeat))
        downbeat_indices.append(idx)
    return downbeat_indices


def top_p(logits, thres=0.9, temperature=1.0):
    assert logits.dim() == 2, logits.shape

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 0] = False
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9):
    assert logits.dim() == 2

    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs
