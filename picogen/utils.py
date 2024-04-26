import pickle
import torch
import torch.nn.functional as F
import shutil
import math
import questionary
from dataclasses import dataclass, fields
import json

def pickle_load(file):
    return pickle.load(open(file, 'rb'))

def pickle_save(data, file):
    pickle.dump(data, open(file, 'wb'))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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
    d_model: int
    num_layers: int
    num_heads: int
    activation: str
    dropout: float
    max_seq_len: int
    max_position_embeddings: int
    cp_word_size: int

def load_config(config_file, verbose=False):
    config = json.loads(config_file.read_text())
    hp = HyperParam(**config)
    if verbose:
        print('checkpoint model config:')
        for v in fields(hp):
            print(f'\t{v.name}: {getattr(hp, v.name)}')
    return hp

def query_mkdir(path):
    if not path.exists():
        confirm = questionary.confirm(f'Directory "{path}" does not exist, create?', default=False).ask()
        if confirm:
            path.mkdir(parents=True)
        else:
            print(f'Directory "{path}" does not exist. Exit.')
            exit(1)

def init_ckpt_dir(ckpt_dir, config_file, config_name = 'config'):
    t_path = (ckpt_dir / config_name).with_suffix(config_file.suffix)
    if not t_path.exists():
        # ckpt_dir.mkdir(exist_ok=True)
        query_mkdir(ckpt_dir)
        shutil.copyfile(config_file, t_path)
    else:
        # check if config is the same
        if config_file.read_text() != t_path.read_text():
            override = questionary.confirm(
                f'Config file "{config_file}" is not same with checkpoint "{t_path}", override?',
                default=False,
            ).ask()
            if override:
                shutil.copyfile(config_file, t_path)
            else:
                print('Conflicting config file: "{}" and "{}". Exit.'.format(config_file, t_path))
                exit(1)
                # raise ValueError(f'config file {config_file} and {t_path} are not the same')

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

def scan_checkpoint(cp_dir, prefix):
    # pattern = os.path.join(cp_dir, prefix + '????????')
    # cp_list = glob.glob(pattern)
    cp_list = list(cp_dir.glob(f'{prefix}*'))
    if len(cp_list) == 0:
        return None
    return sorted(cp_list, key=lambda n: int(n.stem.split("_")[-1]))[-1]

def load_checkpoint(filepath, device):
    assert filepath.is_file()
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def top_p(logits, thres = 0.9, temperature = 1.0):
    assert logits.dim() == 2

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # sorted_logits = sorted_logits / temperature
    cum_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)

    # sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    # sorted_logits = sorted_logits * temperature
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    assert logits.dim() == 2

    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
