from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPTNeoXConfig, GPTNeoXModel

from .utils import top_p


def _get_device(module):
    return next(module.parameters()).device


class ConditionEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.l1_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hp.d_model,
                nhead=hp.num_heads,
                dim_feedforward=hp.d_model * 4,
                dropout=hp.dropout,
                activation=hp.activation,
                batch_first=True,
            ),
            hp.num_layers_encoder,
        )
        self.pos_emb = nn.Embedding(hp.condition_class, hp.d_model)
        self.bottlenect = nn.Sequential(
            nn.Linear(hp.d_model, hp.d_bottleneck),
            nn.ReLU(),
            nn.Linear(hp.d_bottleneck, hp.d_model),
        )

    def forward(self, input_embs):
        B, L, N, D = input_embs.shape
        pos = torch.arange(N).to(input_embs.device)
        pos = self.pos_emb(pos)[None, None, :, :].expand(B, L, N, D)
        input_embs = input_embs + pos
        out = self.l1_encoder(input_embs.view(B * L, N, D)).view(B, L, N, D)
        out = out[:, :, 0, :]
        assert out.shape == (B, L, D)
        out = self.bottlenect(out)
        return out


class PiCoGenDecoder(nn.Module):
    class InputClass(Enum):
        TARGET = 0
        CONDITION = 1

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        config = GPTNeoXConfig(
            vocab_size=hp.vocab_size,
            hidden_size=hp.d_model,
            num_hidden_layers=hp.num_layers,
            num_attention_heads=hp.num_heads,
            intermediate_size=hp.d_model * 4,
            hidden_act=hp.activation,
            hidden_dropout=hp.dropout,
            max_position_embeddings=hp.max_position_embeddings,
        )
        self.model = GPTNeoXModel(config)
        self.word_emb = nn.Embedding(hp.vocab_size, hp.d_model, padding_idx=0)
        self.cond_encoder = ConditionEncoder(hp)
        self.cls_emb = nn.Embedding(
            hp.token_class, hp.d_model, padding_idx=0
        )  # 0: target, 1: condition
        self.lm_head = nn.Linear(hp.d_model, hp.vocab_size)

    def generate(
        self, input_seg, input_cls_ids, need_encode, kv_cache=None, temperature=1.0, thres=0.9
    ):
        B, L = input_cls_ids.shape

        if kv_cache is None:
            input_ids = torch.zeros(B, L, device=_get_device(self.word_emb)).long()
            input_cond_embs = torch.zeros(
                B,
                L,
                self.hp.condition_class,
                self.hp.d_model,
                device=_get_device(self.cond_encoder),
            ).float()
            for b in range(B):
                for ll in range(L):
                    if need_encode[b, ll]:
                        emb = torch.FloatTensor(np.array(input_seg[b][ll])).to(
                            _get_device(self.cond_encoder)
                        )
                        input_cond_embs[b, ll] = emb
                    else:
                        input_ids[b, ll] = input_seg[b][ll]
        else:  # NOTE: only use the last token as input
            input_ids = torch.zeros(B, 1, device=_get_device(self.word_emb)).long()
            input_cond_embs = torch.zeros(
                B,
                1,
                self.hp.condition_class,
                self.hp.d_model,
                device=_get_device(self.cond_encoder),
            ).float()
            for b in range(B):
                if need_encode[b, -1]:
                    emb = torch.FloatTensor(np.array(input_seg[b][-1])).to(
                        _get_device(self.cond_encoder)
                    )
                    input_cond_embs[b, -1] = emb
                else:
                    input_ids[b, -1] = input_seg[b][-1]
            input_cls_ids = input_cls_ids[:, -1:]
        assert input_ids.shape == input_cls_ids.shape

        input_embs = self.word_emb(input_ids)
        input_cond_embs = self.cond_encoder(input_cond_embs)
        input_cls_embs = self.cls_emb(input_cls_ids)

        if kv_cache is None:
            mask = (input_embs.sum(dim=-1, keepdim=True) != 0).expand(B, L, self.hp.d_model)
        else:
            mask = (input_embs.sum(dim=-1, keepdim=True) != 0).expand(B, 1, self.hp.d_model)
        input_cond_embs[mask] = 0  # NOTE: where input_embs is not zero

        input_embs = input_embs + input_cond_embs + input_cls_embs

        model_out = self.model(
            inputs_embeds=input_embs,
            past_key_values=kv_cache,
        )

        logits = self.lm_head(model_out.last_hidden_state)[:, -1, :]
        assert logits.shape == (B, self.hp.vocab_size)
        probs = F.softmax(top_p(logits, thres=thres, temperature=temperature), dim=-1)
        output_ids = torch.multinomial(probs, num_samples=1)
        assert output_ids.shape == (B, 1)

        return output_ids, model_out.past_key_values

    def forward(
        self,
        input_seqs,
        input_cls_ids,
        need_encode,
        input_ids=None,
        input_cond_embs=None,
        labels=None,
        kv_cache=None,
    ):
        B, L = input_cls_ids.shape
        input_cls_ids = input_cls_ids.to(_get_device(self.cls_emb))

        if input_seqs is not None:
            assert input_ids is None and input_cond_embs is None
            input_ids = torch.zeros(B, L, device=_get_device(self.word_emb)).long()
            input_cond_embs = torch.zeros(
                B,
                L,
                self.hp.condition_class,
                self.hp.d_model,
                device=_get_device(self.cond_encoder),
            ).float()
            for b in range(B):
                for ll in range(L):
                    if need_encode[b, ll]:
                        emb = torch.FloatTensor(np.array(input_seqs[b][ll])).to(
                            _get_device(self.cond_encoder)
                        )
                        input_cond_embs[b, ll] = emb
                    else:
                        input_ids[b, ll] = input_seqs[b][ll]
        else:
            assert input_ids is not None and input_cond_embs is not None
            input_ids = input_ids.to(_get_device(self.word_emb))
            input_cond_embs = input_cond_embs.to(_get_device(self.cond_encoder))

        input_embs = self.word_emb(input_ids)
        input_cond_embs = self.cond_encoder(input_cond_embs)
        input_cls_embs = self.cls_emb(input_cls_ids)

        mask = (input_embs.sum(dim=-1, keepdim=True) != 0).expand(B, L, self.hp.d_model)
        input_cond_embs[mask] = 0  # NOTE: where input_embs is not zero

        input_embs = input_embs + input_cond_embs + input_cls_embs

        model_out = self.model(
            inputs_embeds=input_embs,
            past_key_values=kv_cache,
        )

        logits = self.lm_head(model_out.last_hidden_state)
        assert logits.shape == (B, L, self.hp.vocab_size)

        lm_loss = None
        if labels is not None:
            assert labels.shape == (B, L)
            labels = labels.to(logits.device)

            loss_fct = F.cross_entropy
            lm_loss = loss_fct(logits.view(-1, self.hp.vocab_size), labels.view(-1))

        out = {
            "loss": lm_loss,
            "logits": logits,
            "past_key_values": model_out.past_key_values,
            "hidden_states": model_out.hidden_states,
            "attentions": model_out.attentions,
        }

        return out
