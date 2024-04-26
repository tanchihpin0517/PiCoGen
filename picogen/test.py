import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import json

from .model import CPTransformer
from .data import AILabs1k7Dataset
# from .repr import BarEvent, SpecEvent, Tokenizer
from .repr import Event, Tokenizer
from .config import YamlConfig
from .utils import AttrDict, scan_checkpoint, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_use_cache', action="store_true", help='for test')
    parser.add_argument('--debug', action="store_true", help='for test')
    # parser.add_argument('--accel', type=str, default='gpu', help='gpu or cpu or mps (for mac)')

    # required
    parser.add_argument('--data_dir', type=Path, required=True, help='data directory')
    parser.add_argument('--save_dir', type=Path, required=True, help='where to save results')
    parser.add_argument('--ckpt_dir', type=Path, required=True, help='checkpoint name')
    parser.add_argument('--ckpt_name', type=str, required=True, help='checkpoint name')
    parser.add_argument('--ckpt_step', type=int, required=True, help='steps of checkpoint file')
    parser.add_argument('--vocab_file', type=Path, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    print("command line arguments:")
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))

    test(0, args)

def test(rank, ca):
    # s3_dir = f"s3://kks-trc-workspace/cptan/pop2piano/ls2piano/ckpt/{args.ckpt_name}/lightning_logs/version_{args.ckpt_version}/checkpoints/"
    # cache_dir = f"./ckpt/cache/{ca.ckpt_name}/version_{ca.ckpt_version}"
    # subprocess.run(f"aws s3 sync {s3_dir} {cache_dir} --exclude \"*\" --include \"epoch={args.ckpt_epoch}*\"", shell=True, check=True)

    # ckpt_file = list(Path(cache_dir).glob(f'epoch={ca.ckpt_epoch:03d}*.ckpt'))
    # assert len(ckpt_file) == 1
    # ckpt_file = ckpt_file[0]

    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')

    config_file = ca.ckpt_dir / ca.ckpt_name / "config.json"
    ckpt_file = ca.ckpt_dir / ca.ckpt_name / "models" /  f'model_{ca.ckpt_step:08d}'
    hp = AttrDict(json.loads(config_file.read_text()))
    model = CPTransformer(hp).to(device)
    state_dict = load_checkpoint(ckpt_file, device)
    model.load_state_dict(state_dict['model'])

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    print('checkpoint model config:')
    for key in model.hp:
        print(f'\t{key}: {model.hp[key]}')
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    save_dir = ca.save_dir / f'{ca.ckpt_name}/{ca.ckpt_name}/model_{ca.ckpt_step:08d}'
    save_dir.mkdir(exist_ok=True, parents=True)

    # assert Path(args.ckpt_dir).exists(), f'{args.ckpt_dir} does not exist'

    print('loading data')
    tokenizer = Tokenizer(ca.vocab_file, hp.beat_div, hp.ticks_per_beat)
    dataset = AILabs1k7Dataset(ca.data_dir, tokenizer)
    # trainset = torch.utils.data.Subset(dataset, range(len(dataset)//20, len(dataset)))
    validset = torch.utils.data.Subset(dataset, range(len(dataset)//20))

    print('generating:')
    model.eval()
    model.cuda()
    pbar = tqdm(total=len(validset))
    for i, song in enumerate(validset):
        ls_events = song['ls_events']
        ls_ids = song['ls_ids']
        tgt_seg = [tokenizer.e2i(Event(family='spec', spec='bos'))]
        last_past_kv = None
        family_mask = tokenizer.get_family_mask().to(device)
        out_events = []

        end_event = Event(family='spec', spec='se')
        bar_src_event = Event(family='bar', bar='src')

        for ls_bar_start, ls_bar_end in tokenizer.get_bar_ranges(ls_events):
            tgt_seg.extend(ls_ids[ls_bar_start:ls_bar_end])
            if ls_bar_start == 0:
                tgt_seg.append(tokenizer.e2i(Event(family='spec', spec='ss')))

            # # generate kv cache
            # input_ids = torch.LongTensor(tgt_seg)[None, :, :].to(device)
            # output_ids, past_kv = model.generate(input_ids, family_mask, last_past_kv)
            # last_past_kv = past_kv

            tgt_seg.append(tokenizer.e2i(Event(family='bar', bar='tgt')))

            # tgt_events = [tokenizer.i2e(t) for t in tgt_seg]
            # print(*tgt_events, sep="\n")

            while True:
                # input_ids = torch.LongTensor(tgt_seg)[None, -1:, :].to(device)
                # input_ids = torch.LongTensor(tgt_seg[-1])[None, None, :].to(device)
                input_ids = torch.LongTensor(tgt_seg)[None, :, :].to(device)
                # assert input_ids.shape == (1, 1, len(tgt_seg[-1]))

                output_ids, _ = model.generate(input_ids, family_mask, last_past_kv)
                out_id = output_ids[0][-1].tolist()
                out_event = tokenizer.i2e(out_id)

                out_events.append(out_event)

                if out_event in (bar_src_event, end_event):
                    break
                if len(tgt_seg) > 1024:
                    break


                tgt_seg.append(out_id)
                # last_past_kv = past_kv

                pbar.set_description(f"length: {len(tgt_seg)}")

            if len(tgt_seg) > 1024:
                break

        # tgt_seg = [tokenizer.e2i(Event(family='spec', spec='bos'))]
        # end_event = Event(family='spec', spec='se')
        # last_past_kv = None
        # family_mask = tokenizer.get_family_mask().to(device)
        # out_events = []
        #
        # while True:
        #     input_ids = torch.LongTensor(tgt_seg)[None, -1:, :].to(device)
        #     output_ids, past_kv = model.generate(input_ids, family_mask, last_past_kv)
        #     out_id = output_ids[0][-1].tolist()
        #     out_event = tokenizer.i2e(out_id)
        #
        #     out_events.append(out_event)
        #
        #     if out_event == end_event:
        #         break
        #     if len(tgt_seg) > 1024:
        #         break
        #
        #
        #     tgt_seg.append(out_id)
        #     last_past_kv = past_kv
        #
        #     pbar.set_description(f"length: {len(tgt_seg)}")
        #
        # print(*out_events, sep="\n")

        tokenizer.events_to_midi(
            out_events,
            song['metadata']['beat_per_bar'],
        ).dump(ca.save_dir / f'{i}.mid')

        # prefix = f"{Path(data['source'][0]).stem}"
        # out_dir = save_dir / prefix
        # out_dir.mkdir(exist_ok=True, parents=True)
        #
        # ls_events = tokenizer.ids_to_events(data['ls_ids'][0].tolist())
        # (out_dir / f"{prefix}_ls.txt").write_text("\n".join(map(str, ls_events)))
        # ls_midi = tokenizer.events_to_midi(ls_events)
        # ls_midi.dump(out_dir / f"{prefix}_ls.mid")
        #
        # out_events = tokenizer.ids_to_events(seq_out[0].tolist())
        # (out_dir / f"{prefix}_test.txt").write_text("\n".join(map(str, out_events)))
        # out_midi = tokenizer.events_to_midi(out_events)
        # out_midi.dump(out_dir / f"{prefix}_test.mid")
        #
        # pbar.update(1)
        # continue
        # exit()


        # data['m_ids'] = data['m_ids'].tolist()
        # data['ls_ids'] = data['ls_ids'].tolist()
        # seq_out = seq_out.tolist()
        # convert_fn = utils.remi_to_midi if model_config['data']['repr'] == 'remi' else utils.cp_to_midi
        #
        # def convert_type(inp):
        #     if model_config['data']['repr'] == 'cp':
        #         for i, seq in enumerate(inp[0]):
        #             if seq[0] != 3:
        #                 inp[0][i] = list(map(int, seq))
        #             else:
        #                 inp[0][i][0] = int(seq[0])
        #                 inp[0][i][1] = round(seq[1], 2)
        #                 for j in range(2, len(seq)):
        #                     inp[0][i][j] = int(seq[j])
        #     else:
        #         inp[0] = list(map(int, inp[0]))
        #     return inp
        #
        # data['m_ids'] = convert_type(data['m_ids'])
        # data['ls_ids'] = convert_type(data['ls_ids'])
        # seq_out = convert_type(seq_out)
        #
        # orig = [vocab[t] for t in data['m_ids'][0]]
        # (save_dir / f"{prefix}_orig.txt").write_text("\n".join(map(str, orig)))
        # orig_midi = convert_fn(
        #     orig,
        #     beat_per_bar=4,
        #     grid_tick=model_config['data']['grid_tick'],
        #     grid_div=model_config['data']['grid_div'],
        # )
        # orig_midi.dump(save_dir / f"{prefix}_orig.mid")
        #
        # ls = [vocab[t] for t in data['ls_ids'][0]]
        # (save_dir / f"{prefix}_ls.txt").write_text("\n".join(map(str, ls)))
        # ls_midi = convert_fn(
        #     ls,
        #     beat_per_bar=4,
        #     grid_tick=model_config['data']['grid_tick'],
        #     grid_div=model_config['data']['grid_div'],
        # )
        # ls_midi.dump(save_dir / f"{prefix}_ls.mid")
        #
        # out = [vocab[t] for t in seq_out[0]]
        # (save_dir / f"{prefix}_test.txt").write_text("\n".join(map(str, out)))
        # out_midi = convert_fn(
        #     out,
        #     beat_per_bar=4,
        #     grid_tick=model_config['data']['grid_tick'],
        #     grid_div=model_config['data']['grid_div'],
        # )
        # out_midi.dump(save_dir / f"{prefix}_test.mid")


        pbar.update(1)
    pbar.close()

        # print(seq_out_start.shape)
        # print(out.shape)
        # for t in out[0].tolist():
        #     print(vocab[t])



if __name__ == '__main__':
    main()

