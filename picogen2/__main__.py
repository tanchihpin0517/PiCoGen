import argparse
from pathlib import Path

import questionary

from .data import download
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

    prepare_parser = subparsers.add_parser("prepare", help="Prepare the training data")
    prepare_parser.set_defaults(func=command_prepare)
    prepare_parser.add_argument("--data_dir", type=Path, required=True)
    prepare_parser.add_argument("--bt_ckpt_path", type=Path, required=True)
    prepare_parser.add_argument("--loglevel", type=str, default="INFO")
    prepare_parser.add_argument("--n_jobs", type=int, default=1)
    prepare_parser.add_argument("--n_procs_per_gpu", type=int, default=1)
    prepare_parser.add_argument("--n_data", type=int)
    prepare_parser.add_argument("--debug", action="store_true")
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--part_num", type=int, default=1)
    prepare_parser.add_argument("--part", type=int, default=0)
    prepare_parser.add_argument("--regen_available_pairs", action="store_true")
    prepare_parser.add_argument("tasks", nargs="*")

    args = parser.parse_args()
    logger.setLevel(args.loglevel)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def command_download(args):
    if not args.data_dir.exists():
        if questionary.confirm(f"Create directory {args.data_dir}?").ask():
            args.data_dir.mkdir(parents=True)
        else:
            logger.error(f"Directory {args.data_dir} does not exist")

    download.pop2piano(args.song_file, args.data_dir, partition=args.partition)


def command_prepare(args):
    pass


if __name__ == "__main__":
    main()
