import argparse
import json
from pathlib import Path

import questionary

from . import infer
from .data import download, preprocess
from .repr import Vocab, gen_vocab
from .utils import logger


def main():
    parser = argparse.ArgumentParser(description="The training script for PiCoGen2")
    subparsers = parser.add_subparsers()

    download_parser = subparsers.add_parser(
        "download", help="Download Pop2Piano dataset from YouTube"
    )
    download_parser.set_defaults(func=command_download)
    download_parser.add_argument("--song_file", type=Path, required=True)
    download_parser.add_argument("--data_dir", type=Path, required=True)
    download_parser.add_argument("--loglevel", type=str, default="WARN")
    download_parser.add_argument("-p", "--partition", type=int, nargs=2, default=(0, 1))

    preprocess_parser = subparsers.add_parser("preprocess", help="Prepare the training data")
    preprocess_parser.set_defaults(func=command_preprocess)
    preprocess_parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "transcribe", "beat", "sheetsage"],
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

    vocab_parser = subparsers.add_parser("generate_vocab", help="Generate vocabulary file")
    vocab_parser.set_defaults(func=command_vocab)
    vocab_parser.add_argument("--output_file", type=Path, required=True)

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
        infer.download(args.input_url, args.output_dir / "song.mp3")

    if args.stage == "beat":
        infer.detect_beat(args.input_audio, args.output_dir / "song_beat.json")

    if args.stage == "sheetsage":
        infer.extract_sheetsage_feature(
            args.input_audio,
            args.output_dir / "song_sheetsage.npz",
            args.output_dir / "song_beat.json",
        )

    if args.stage == "piano":
        infer.picogen2(
            beat_file=args.output_dir / "song_beat.json",
            sheetsage_file=args.output_dir / "song_sheetsage.npz",
            output_dir=args.output_dir,
            config_file=args.config_file,
            vocab_file=args.vocab_file,
            ckpt_file=args.ckpt_file,
            max_bar_num=args.max_bar_num,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
