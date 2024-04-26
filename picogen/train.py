import time
import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from pathlib import Path
from torch.nn import functional as F

from . import utils
from .utils import AttrDict, scan_checkpoint, save_checkpoint, load_checkpoint, load_config
from .model import CPTransformer
from .repr import Event, Tokenizer
from .data import AILabs1k7Dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--config_file', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--vocab_file', type=Path, required=True)
    parser.add_argument('--training_epochs', default=1000, type=int)
    parser.add_argument('--training_steps', default=150000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--stdout_interval', default=1, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)

    return parser.parse_args()

def main():
    ca = parse_args()

    print('Initializing Training Process..')
    hp = load_config(ca.config_file, verbose=True)
    utils.init_ckpt_dir(ca.checkpoint_path, ca.config_file)

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        print('Batch size per GPU :', ca.batch_size)
    else:
        pass

    train(ca, hp)

def train(ca, hp):
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(0))
    else:
        device = torch.device('cpu')

    model = CPTransformer(hp).to(device)

    assert ca.checkpoint_path.is_dir()
    model_dir = ca.checkpoint_path / "models"
    log_dir = ca.checkpoint_path / "logs"
    valid_dir = ca.checkpoint_path / "validations"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    print(model)
    print("checkpoints directory : ", ca.checkpoint_path)
    ckpt = scan_checkpoint(model_dir, 'model_')

    steps = 0
    if ckpt is None:
        state_dict = None
        last_epoch = -1
        last_batch = -1
    else:
        state_dict = load_checkpoint(ckpt, device)
        model.load_state_dict(state_dict['model'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        last_batch = state_dict['batch']

    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), hp.learning_rate, betas=[hp.adam_b1, hp.adam_b2])
    if state_dict is not None:
        optim.load_state_dict(state_dict['optim'])

    s1 = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1/hp.warmup_epochs,
        total_iters=hp.warmup_epochs,
        last_epoch=last_epoch
    )
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=hp.sched_T,
        eta_min=hp.learning_rate_min,
        last_epoch=last_epoch
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[s1, s2],
        milestones=[hp.warmup_epochs],
        last_epoch=last_epoch
    )

    print("Loading dataset...")
    print("Vocab file : ", ca.vocab_file)
    print("Data directory : ", ca.data_dir)
    tokenizer = Tokenizer(ca.vocab_file, hp.beat_div, hp.ticks_per_beat)
    dataset = AILabs1k7Dataset(ca.data_dir, tokenizer)
    trainset = torch.utils.data.Subset(dataset, range(len(dataset) // 20, len(dataset)))
    # validset = torch.utils.data.Subset(dataset, range(len(dataset) // 20))

    train_loader = AILabs1k7Dataset.get_dataloader(
        tokenizer,
        hp.max_seq_len,
        trainset,
        num_workers=mp.cpu_count(),
        batch_size=ca.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    sw = SummaryWriter(log_dir)

    model.train()
    for epoch in range(max(0, last_epoch), ca.training_epochs):
        start_e = time.time()
        print("Epoch: {}".format(epoch+1))

        pbar = tqdm(desc=f'Epoch {epoch+1}', total=len(trainset), dynamic_ncols=True)
        for batch_i, batch in enumerate(train_loader):
            tgt_ids = batch['tgt_ids'].to(device)
            label_ids = batch['label_ids'].to(device)

            if last_batch >= 0:
                last_batch -= 1
                steps += 1
                pbar.update(tgt_ids.shape[0])
                continue

            out = model(
                input_ids = tgt_ids,
                labels = label_ids,
            )
            optim.zero_grad()
            loss = out['loss']
            loss.backward()
            optim.step()

            if steps % ca.stdout_interval == 0:
                pbar.set_postfix_str(f'step={steps}, loss={loss.item():4.3f}')

            # checkpointing
            if steps % ca.checkpoint_interval == 0 and steps != 0:
                if steps % ca.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = model_dir / f'model_{steps:08d}'
                    save_checkpoint(
                        checkpoint_path, {
                            'model': (model.module if ca.num_gpus > 1 else model).state_dict(),
                            'optim': optim.state_dict(),
                            'steps': steps,
                            'epoch': epoch,
                            'batch': batch_i,
                        }
                    )

            # Tensorboard summary logging
            if steps % ca.summary_interval == 0:
                sw.add_scalar("training/loss", loss.item(), steps)

            steps += 1
            pbar.update(tgt_ids.shape[0])

            if steps > ca.training_steps:
                break

        pbar.close()
        if steps > ca.training_steps:
            break
        scheduler.step()
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start_e)))

if __name__ == '__main__':
    main()
