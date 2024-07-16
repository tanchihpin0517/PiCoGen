import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPTNeoXConfig, GPTNeoXModel

from .utils import top_p


class CPTransformer(nn.Module):
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
        self.heads = nn.ModuleList()
        for _ in range(hp.cp_word_size):
            self.heads.append(nn.Linear(hp.d_model, hp.vocab_size))

    def generate(self, input_ids, fmask, kv_cache=None):
        B, L, C = input_ids.shape
        assert C == self.hp.cp_word_size

        input_embs = self.word_emb(input_ids)
        input_embs = input_embs.sum(dim=-2)
        assert input_embs.shape == (B, L, self.hp.d_model)

        model_out = self.model(
            inputs_embeds=input_embs,
            past_key_values=kv_cache,
        )

        logits = []
        for head in self.heads:
            logit = head(model_out.last_hidden_state)[:, -1, :]
            logit = logit[:, None, :]
            assert logit.shape == (B, 1, self.hp.vocab_size)
            logits.append(logit)
        logits = torch.cat(logits, dim=-2)
        assert logits.shape == (B, self.hp.cp_word_size, self.hp.vocab_size)

        last_output_ids = []
        for i in range(B):
            # prob = F.softmax((logits[i]), dim = -1)
            prob = F.softmax(top_p(logits[i]), dim=-1)
            ids = torch.multinomial(prob, num_samples=1)[:, 0]
            ids = ids * fmask[ids[0]]
            last_output_ids.append(ids)
        last_output_ids = torch.stack(last_output_ids, dim=0)
        assert last_output_ids.shape == (B, self.hp.cp_word_size), last_output_ids.shape

        output_ids = torch.cat([input_ids, last_output_ids[:, None, :]], dim=-2)

        return output_ids, model_out.past_key_values

    def forward(
        self,
        input_ids,
        # input_family_masks,
        labels=None,
        kv_cache=None,
    ):
        B, L, C = input_ids.shape
        assert C == self.hp.cp_word_size

        # input_ids = input_ids * (input_family_masks == 0).long()
        input_embs = self.word_emb(input_ids)
        input_embs = input_embs.sum(dim=-2)
        assert input_embs.shape == (B, L, self.hp.d_model)

        model_out = self.model(
            inputs_embeds=input_embs,
            past_key_values=kv_cache,
        )

        logits = []
        for head in self.heads:
            logit = head(model_out.last_hidden_state)
            logit = logit[:, :, None, :]
            assert logit.shape == (B, L, 1, self.hp.vocab_size)
            logits.append(logit)
        logits = torch.cat(logits, dim=-2)
        assert logits.shape == (B, L, self.hp.cp_word_size, self.hp.vocab_size)

        lm_loss = None
        if labels is not None:
            assert labels.shape == (B, L, self.hp.cp_word_size)
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            # shift_logits = logits[:, :-1, :, :].contiguous()
            # labels = labels[:, 1:, :].contiguous()
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


# class MIDITransformer(nn.Module):
#     def __init__(self, hp):
#         super().__init__()
#         self.hp = hp
#         config = GPTNeoXConfig(
#             vocab_size = hp.vocab_size,
#             hidden_size = hp.d_model,
#             num_hidden_layers = hp.num_layers,
#             num_attention_heads = hp.num_heads,
#             intermediate_size = hp.d_model * 4,
#             hidden_act = hp.activation,
#             hidden_dropout = hp.dropout,
#             max_position_embeddings = hp.max_position_embeddings,
#         )
#         self.model = GPTNeoXModel(config)
#         self.word_emb = nn.Embedding(hp.vocab_size, hp.d_model, padding_idx=0)
#         self.heads = nn.ModuleList()
#         for _ in range(hp.cp_word_size):
#             self.heads.append(nn.Linear(hp.d_model, hp.vocab_size))
#
#     def generate(self, input_ids, fmask, kv_cache = None):
#         B, L, C = input_ids.shape
#         assert C == self.hp.cp_word_size
#
#         input_embs = self.word_emb(input_ids)
#         input_embs = input_embs.sum(dim = -2)
#         assert input_embs.shape == (B, L, self.hp.d_model)
#
#         model_out = self.model(
#             inputs_embeds = input_embs,
#             past_key_values = kv_cache,
#         )
#
#         logits = []
#         for head in self.heads:
#             logit = head(model_out.last_hidden_state)[:, -1, :]
#             logit = logit[:, None, :]
#             assert logit.shape == (B, 1, self.hp.vocab_size)
#             logits.append(logit)
#         logits = torch.cat(logits, dim = -2)
#         assert logits.shape == (B, self.hp.cp_word_size, self.hp.vocab_size)
#
#         last_output_ids = []
#         for i in range(B):
#             #prob = F.softmax((logits[i]), dim = -1)
#             prob = F.softmax(top_p(logits[i]), dim = -1)
#             ids = torch.multinomial(prob, num_samples = 1)[:, 0]
#             ids = ids * fmask[ids[0]]
#             last_output_ids.append(ids)
#         last_output_ids = torch.stack(last_output_ids, dim = 0)
#         assert last_output_ids.shape == (B, self.hp.cp_word_size), last_output_ids.shape
#
#         output_ids = torch.cat([input_ids, last_output_ids[:, None, :]], dim = -2)
#
#         return output_ids, model_out.past_key_values
#
#     def forward(
#         self,
#         input_ids,
#         # input_family_masks,
#         labels = None,
#         kv_cache = None
#     ):
#         B, L, C = input_ids.shape
#         assert C == self.hp.cp_word_size
#
#         # input_ids = input_ids * (input_family_masks == 0).long()
#         input_embs = self.word_emb(input_ids)
#         input_embs = input_embs.sum(dim = -2)
#         assert input_embs.shape == (B, L, self.hp.d_model)
#
#         model_out = self.model(
#             inputs_embeds = input_embs,
#             past_key_values = kv_cache,
#         )
#
#         logits = []
#         for head in self.heads:
#             logit = head(model_out.last_hidden_state)
#             logit = logit[:, :, None, :]
#             assert logit.shape == (B, L, 1, self.hp.vocab_size)
#             logits.append(logit)
#         logits = torch.cat(logits, dim = -2)
#         assert logits.shape == (B, L, self.hp.cp_word_size, self.hp.vocab_size)
#
#         lm_loss = None
#         if labels is not None:
#             assert labels.shape == (B, L, self.hp.cp_word_size)
#             # move labels to correct device to enable model parallelism
#             labels = labels.to(logits.device)
#             # we are doing next-token prediction; shift prediction scores and input ids by one
#             # shift_logits = logits[:, :-1, :, :].contiguous()
#             # labels = labels[:, 1:, :].contiguous()
#             loss_fct = F.cross_entropy
#             lm_loss = loss_fct(logits.view(-1, self.hp.vocab_size), labels.view(-1))
#
#         out = {
#             'loss': lm_loss,
#             'logits': logits,
#             'past_key_values': model_out.past_key_values,
#             'hidden_states': model_out.hidden_states,
#             'attentions': model_out.attentions,
#         }
#
#         return out
