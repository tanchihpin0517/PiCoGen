import argparse
import json
from pathlib import Path

import numpy as np
import questionary
import torch

from . import assets, infer
from .data import download, preprocess
from .model import PiCoGenDecoder
from .repr import Tokenizer, Vocab, gen_vocab
from .utils import logger


def main():
    parser = argparse.ArgumentParser(description="The training script for PiCoGen2")
    subparsers = parser.add_subparsers()

    # Download
    download_parser = subparsers.add_parser(
        "download", help="Download Pop2Piano dataset from YouTube"
    )
    download_parser.set_defaults(func=command_download)
    download_parser.add_argument("--song_file", type=Path, required=True)
    download_parser.add_argument("--data_dir", type=Path, required=True)
    download_parser.add_argument("--loglevel", type=str, default="WARN")
    download_parser.add_argument("-p", "--partition", type=int, nargs=2, default=(0, 1))

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Prepare the training data")
    preprocess_parser.set_defaults(func=command_preprocess)
    preprocess_parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=[
            "all",
            preprocess.TASK_TRANS,
            preprocess.TASK_BEAT,
            preprocess.TASK_SHEETSAGE,
            preprocess.TASK_ALIGN,
        ],
        help="Subtask of preprocessing",
    )
    preprocess_parser.add_argument("--data_dir", type=Path, required=True, help="Dataset directory")
    preprocess_parser.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory of saving processed data"
    )
    preprocess_parser.add_argument("--loglevel", type=str, default="INFO")
    preprocess_parser.add_argument("-p", "--partition", type=int, nargs=2, default=(0, 1))
    preprocess_parser.add_argument("--debug", action="store_true")
    preprocess_parser.add_argument("--overwrite", action="store_true")

    # Generate vocabulary
    vocab_parser = subparsers.add_parser("generate_vocab", help="Generate vocabulary file")
    vocab_parser.set_defaults(func=command_vocab)
    vocab_parser.add_argument("--output_file", type=Path, required=True)

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Inference with trained model")
    infer_parser.set_defaults(func=command_infer)
    infer_parser.add_argument(
        "--stage", type=str, choices=["download", "beat", "sheetsage", "piano"], required=True
    )
    infer_parser.add_argument("--input_audio", type=Path, help="input audio file")
    infer_parser.add_argument("--input_url", type=str, help="input audio url")
    infer_parser.add_argument("--loglevel", type=str, default="INFO")
    infer_parser.add_argument("--output_dir", type=Path, required=True)
    infer_parser.add_argument("--config_file", type=Path)
    infer_parser.add_argument("--vocab_file", type=Path)
    infer_parser.add_argument("--ckpt_file", type=Path)
    infer_parser.add_argument("--max_bar_num", type=int)
    infer_parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    if hasattr(args, "loglevel"):
        logger.setLevel(args.loglevel)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def command_download(args):
    if not args.data_dir.exists():
        if questionary.confirm(
            f"The save directory {args.data_dir} does not exist. Create it?"
        ).ask():
            args.data_dir.mkdir(parents=True)
        else:
            logger.error(f"{args.data_dir} does not exist")
            exit(1)

    download.pop2piano(args.song_file, args.data_dir, partition=args.partition)


def command_preprocess(args):
    if args.data_dir == args.output_dir:
        logger.error("data_dir and output_dir should be different")
        exit(1)
    if not args.output_dir.exists():
        if questionary.confirm(
            f"The output directory {args.output_dir} does not exist. Create it?"
        ).ask():
            args.output_dir.mkdir(parents=True)
        else:
            logger.error(f"{args.output_dir} does not exist")
            exit(1)

    preprocess.pop2piano(
        args.task,
        args.data_dir,
        args.output_dir,
        partition=args.partition,
        debug=args.debug,
        overwrite=args.overwrite,
    )


def command_vocab(args):
    vocab = gen_vocab()
    args.output_file.write_text(json.dumps(vocab, indent=4))
    vocab = Vocab(args.output_file)
    print(vocab)
    print("vocab size:", vocab.len())


def command_infer(args):
    if not args.output_dir.exists():
        if questionary.confirm(
            f"The output directory {args.output_dir} does not exist. Create it?"
        ).ask():
            args.output_dir.mkdir(parents=True)
        else:
            logger.error(f"{args.output_dir} does not exist")
            exit(1)

    if args.stage == "download":
        logger.info(f"Downloading {args.input_url}")
        infer.download(args.input_url, args.output_dir / "song.mp3")

    if args.stage == "beat":
        logger.info(f"Detecting beat for {args.input_audio}")
        infer.detect_beat(args.input_audio, args.output_dir / "song_beat.json")

    if args.stage == "sheetsage":
        logger.info(f"Extracting SheetSage feature for {args.input_audio}")
        infer.extract_sheetsage_feature(
            args.input_audio,
            args.output_dir / "song_sheetsage.npz",
            args.output_dir / "song_beat.json",
        )

    if args.stage == "piano":
        assert args.output_dir.is_dir(), f"{args.output_dir} is not a directory"

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        ckpt_file = assets.checkpoint_file() if args.ckpt_file is None else args.ckpt_file
        vocab_file = assets.vocab_file() if args.vocab_file is None else args.vocab_file
        beat_file = args.output_dir / "song_beat.json"
        sheetsage_file = args.output_dir / "song_sheetsage.npz"

        logger.info(f"Loading model from {ckpt_file}")
        model = PiCoGenDecoder.from_pretrained(device=device)
        logger.info("Number of parameters: {}".format(sum(p.numel() for p in model.parameters())))

        logger.info(f"Loading tokenizer from {vocab_file}")
        tokenizer = Tokenizer()

        logger.info(f"Loading beat information from {beat_file}")
        beat_information = json.loads(beat_file.read_text())

        logger.info(f"Loading SheetSage's last hidden state from {sheetsage_file}")
        sheetsage_last_hidden_state = np.load(sheetsage_file)
        melody_last_embs = sheetsage_last_hidden_state["melody"]
        harmony_last_embs = sheetsage_last_hidden_state["harmony"]

        logger.info("Generating piano cover")
        out_events = infer.decode(
            model=model,
            tokenizer=tokenizer,
            beat_information=beat_information,
            melody_last_embs=melody_last_embs,
            harmony_last_embs=harmony_last_embs,
            max_bar_num=args.max_bar_num,
            temperature=args.temperature,
        )

        (args.output_dir / "piano.txt").write_text("\n".join(map(str, out_events)))
        tokenizer.events_to_midi(out_events).dump(args.output_dir / "piano.mid")


if __name__ == "__main__":
    main()
