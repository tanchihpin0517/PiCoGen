import time

import torch
import torch.multiprocessing as mp
from torch.distributed import barrier, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import utils
from .data.dataset import Pop2PianoDataset
from .model import PiCoGenDecoder
from .repr import Tokenizer
from .utils import load_checkpoint, logger, save_checkpoint, scan_checkpoint


def main(args):
    logger.info("Initializing Training Process..")
    hp = utils.load_config(args.config_file)
    utils.init_ckpt_dir(args.checkpoint_path, args.config_file)

    logger.info("Hyperparameters:")
    logger.info(hp)

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        args.batch_size = args.batch_size // args.num_gpus
        print("Batch size per GPU :", args.batch_size)
    else:
        pass

    if args.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=args.num_gpus,
            args=(
                args,
                hp,
            ),
        )
    else:
        train(0, args, hp)


def train(rank, ca, hp):
    if ca.num_gpus > 1:
        init_process_group(
            backend=ca.dist_backend,
            init_method=ca.dist_url,
            world_size=ca.world_size * ca.num_gpus,
            rank=rank,
        )
        barrier()

    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
    else:
        device = torch.device("cpu")

    model = PiCoGenDecoder(hp).to(device)

    assert ca.checkpoint_path.is_dir()
    model_dir = ca.checkpoint_path / "models"
    log_dir = ca.checkpoint_path / "logs"
    valid_dir = ca.checkpoint_path / "validations"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(model)
        print("checkpoints directory : ", ca.checkpoint_path)
    ckpt = scan_checkpoint(model_dir, "model_")
    if ckpt is None and ca.source_ckpt_file is not None:
        assert (
            ca.source_ckpt_file.is_file()
        ), f"Source checkpoint file {ca.source_ckpt_file} not found."
        ckpt = ca.source_ckpt_file

    state_dict = None
    steps = 0
    last_epoch = -1
    last_batch = -1
    if ckpt is not None:
        state_dict = load_checkpoint(ckpt, device)
        model.load_state_dict(state_dict["model"])
        if ckpt != ca.source_ckpt_file:  # NOTE: if source ckpt is used, start from 0
            steps = state_dict["steps"] + 1
            last_epoch = state_dict["epoch"]
            last_batch = state_dict["batch"]

    if ca.num_gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=True,
        ).to(device)
    else:
        model.to(device)

    optim = torch.optim.AdamW(model.parameters(), hp.learning_rate, betas=(hp.adam_b1, hp.adam_b2))
    if state_dict is not None:
        optim.load_state_dict(state_dict["optim"])

    s1 = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1 / hp.warmup_epochs,
        total_iters=hp.warmup_epochs,
        last_epoch=last_epoch,
    )
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=hp.sched_T, eta_min=hp.learning_rate_min, last_epoch=last_epoch
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[s1, s2], milestones=[hp.warmup_epochs], last_epoch=last_epoch
    )

    print("Loading dataset...")
    print("Vocab file : ", ca.vocab_file)
    print("Data directory : ", ca.dataset_dir, ca.processed_dir)
    tokenizer = Tokenizer(ca.vocab_file, hp.beat_div, hp.ticks_per_beat)
    dataset = Pop2PianoDataset(ca.dataset_dir, ca.processed_dir, tokenizer)
    trainset = dataset
    validset = None

    assert len(tokenizer.vocab) == hp.vocab_size

    train_sampler = DistributedSampler(trainset) if ca.num_gpus > 1 else None
    train_loader = Pop2PianoDataset.get_dataloader(
        tokenizer,
        hp.max_seq_len,
        trainset,
        fn_name="mix",
        num_workers=ca.batch_size,
        batch_size=ca.batch_size,
        sampler=train_sampler,
        shuffle=False if ca.debug or ca.num_gpus > 1 else True,
        pin_memory=True,
    )

    if rank == 0:
        validation_loader = Pop2PianoDataset.get_dataloader(
            tokenizer,
            hp.max_seq_len,
            validset,
            num_workers=mp.cpu_count(),
            batch_size=1,
            sampler=None,
            shuffle=False,
        )
        sw = SummaryWriter(log_dir)

    model.train()
    for epoch in range(max(0, last_epoch), ca.training_epochs):
        start_e = 0

        if rank == 0:
            start_e = time.time()
            print("Epoch: {}".format(epoch + 1))

        if ca.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(desc=f"Epoch {epoch+1}", total=len(trainset), dynamic_ncols=True)
        for batch_i, batch in enumerate(train_loader):
            batch_piano = batch["piano"]
            input_ids = batch_piano["input_ids"]
            input_cond_embs = batch_piano["input_cond_embs"]
            label_ids = batch_piano["label_ids"]
            input_cls_ids = batch_piano["input_cls_ids"]
            need_encode = batch_piano["need_encode"]

            if hp.loss_weight != 0:
                piano_out = model(
                    input_seqs=None,
                    input_ids=input_ids,
                    input_cond_embs=input_cond_embs,
                    input_cls_ids=input_cls_ids,
                    need_encode=need_encode,
                    labels=label_ids,
                )
            else:
                piano_out = {"loss": torch.tensor(0).to(device)}

            batch_pair = batch["pair"]
            input_ids = batch_pair["input_ids"]
            input_cond_embs = batch_pair["input_cond_embs"]
            label_ids = batch_pair["label_ids"]
            input_cls_ids = batch_pair["input_cls_ids"]
            need_encode = batch_pair["need_encode"]

            if hp.loss_weight != 1:
                pair_out = model(
                    input_seqs=None,
                    input_ids=input_ids,
                    input_cond_embs=input_cond_embs,
                    input_cls_ids=input_cls_ids,
                    need_encode=need_encode,
                    labels=label_ids,
                )
            else:
                pair_out = {"loss": torch.tensor(0).to(device)}

            if pair_out["loss"].isnan().any():
                continue

            optim.zero_grad()
            assert 0 <= hp.loss_weight <= 1
            loss = hp.loss_weight * piano_out["loss"] + (1 - hp.loss_weight) * pair_out["loss"]
            loss.backward()
            optim.step()

            if rank == 0:
                if steps % ca.stdout_interval == 0:
                    pbar.set_postfix_str(f"step={steps}, loss={loss.item():4.3f}")

                # checkpointing
                if steps % ca.checkpoint_interval == 0 and steps != 0:
                    if steps % ca.checkpoint_interval == 0 and steps != 0:
                        checkpoint_path = model_dir / f"model_{steps:08d}"
                        save_checkpoint(
                            checkpoint_path,
                            {
                                "model": (model.module if ca.num_gpus > 1 else model).state_dict(),
                                "optim": optim.state_dict(),
                                "steps": steps,
                                "epoch": epoch,
                                "batch": batch_i,
                            },
                        )

                # Tensorboard summary logging
                if steps % ca.summary_interval == 0:
                    sw.add_scalar("training/loss", loss.item(), steps)

                # Validation
                if steps > 10 * ca.validation_interval and steps % ca.validation_interval == 0:
                    pass

            steps += 1
            if rank == 0:
                pbar.update(len(input_ids) * ca.num_gpus)

            if steps > ca.training_steps:
                break

        if rank == 0:
            pbar.close()
        if steps > ca.training_steps:
            break

        scheduler.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(epoch + 1, int(time.time() - start_e))
            )
