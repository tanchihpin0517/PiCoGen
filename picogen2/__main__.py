import argparse
from pathlib import Path

import questionary

from .data import download, preprocess
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
        choices=["all", "transcribe", "beat"],
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

    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
