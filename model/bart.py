import torch
from .modeing_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math
from .utils import get_span_from_pred, get_ent_boundary
from .multi_head_biaffine3 import MultiHeadBiaffine
from copy import deepcopy

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class FBartDecoder_NER_ONLY(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([0,2]+label_ids) # 改mapping TODO
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits
    
class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([46317, 2, 47114]+label_ids) # 改mapping TODO
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        
        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 3:self.src_start_index] = tag_scores # 这也要改成3
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[:, -1]


class CaGFBartDecoder_NER_ONLY(FBartDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, tokens, state, update_tree=False):
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0) # 为0的地方说明是word，不能从mapping里面取用。
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
 
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        elif update_tree:
            # wxl, eval但是返回所有token的score
            tokens = tokens[:, :-1] # 去掉最后一个eos的输入
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size

        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class

        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size
        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            word_scores = (gen_scores + word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits
        
class CaGFBartDecoder(FBartDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, tokens, state, update_tree=False):
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0) # 为0的地方说明是word，不能从mapping里面取用。
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
 
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        elif update_tree:
            
            # wxl, eval但是返回所有token的score
            tokens = tokens[:, :-1] # 去掉最后一个eos的输入
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size

        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class

        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size
        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            word_scores = (gen_scores + word_scores)/2

        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 3:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits


class YCaGFBartDecoder(FBartDecoder):
    # 带MultiHeadBiaffine的
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        self.dropout_layer = nn.Dropout(0.3)
        self.MultiHeadBiaffine = MultiHeadBiaffine(dim=1024, out=1, n_head=1)

    def forward(self, tokens, state, update_tree=False):
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        pad_mask = tokens.eq(-1)
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0) # 预测原文word的地方为0，tag和eos的地方不变
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数
        tokens_pad = tokens.clone()
        tokens_pad[pad_mask]=-1
        ent_boundary, ent_decode_mask = get_ent_boundary(tokens_pad, 50265)
        if self.training:
            ent_decode_mask = ent_decode_mask[:, :-1]
            decoder_pad_mask = ent_decode_mask.eq(1)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                ent_boundary=ent_boundary,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:ent_decode_mask.size(1), :ent_decode_mask.size(1)],
                                return_dict=True)
        elif update_tree:
            # wxl, eval但是返回所有token的score
            tokens = tokens[:, :-1] # 去掉最后一个eos的输入
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                ent_boundary=ent_boundary,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        #eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        eos_scores = self.MultiHeadBiaffine(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))
        #tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class
        tag_scores = self.MultiHeadBiaffine(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))
        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size
        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
        # word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = self.MultiHeadBiaffine(hidden_state, src_outputs)
        if not self.avg_feature:
            # gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            gen_scores = self.MultiHeadBiaffine(hidden_state, input_embed)
            word_scores = (gen_scores + word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores
        sigmoid = torch.nn.Sigmoid()
        logits = sigmoid(logits)
        return logits


class CaGFBartDecoder_new(FBartDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        self.dropout_layer = nn.Dropout(0.3)
        

    def forward(self, tokens, state, update_tree=False):
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        pad_mask = tokens.eq(-1)
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0) # 预测原文word的地方为0，tag和eos的地方不变
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数
        tokens_pad = tokens.clone()
        tokens_pad[pad_mask]=-1
        ent_boundary, ent_decode_mask = get_ent_boundary(tokens_pad, 50265)
        if self.training:
            ent_decode_mask = ent_decode_mask[:, :-1]
            decoder_pad_mask = ent_decode_mask.eq(1)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                ent_boundary=ent_boundary,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:ent_decode_mask.size(1), :ent_decode_mask.size(1)],
                                return_dict=True)
        elif update_tree:
            # wxl, eval但是返回所有token的score
            tokens = tokens[:, :-1] # 去掉最后一个eos的输入
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                ent_boundary=ent_boundary,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        
        tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class
        
        # 这里有两个融合方式: (1) 特征avg算分数; (2) 各自算分数加起来; 不过这两者是等价的
        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size
        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            
            word_scores = (gen_scores + word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores
        sigmoid = torch.nn.Sigmoid()
        logits = sigmoid(logits)
        return logits


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder
        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        if decoder_type is None:
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type == 'avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=True, use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)


    def prepare_state(self, src_tokens, src_seq_len=None, first=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        return state
    
    
    def forward_self_tgt(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, self_tgt, self_tgt_seq_len, update_tree=False):
        # RE+辅助任务， 暂时不用。
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first)
        decoder_output = self.decoder(tgt_tokens, state, update_tree) # wxl
        decoder_output_self_tgt = self.decoder(self_tgt, state, update_tree) # wxl
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output, 'self_tgt_pred': decoder_output_self_tgt}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0], 'self_tgt_pred': decoder_output_self_tgt[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")
        
    def forward(self, src_tokens, tgt_tokens, re_tgt_tokens, src_seq_len, re_tgt_seq_len, tgt_seq_len, first, update_tree=False):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        src_tokens_clone,  src_seq_len_clone, first_clone = src_tokens.clone(), src_seq_len.clone(), first.clone()
        state = self.prepare_state(src_tokens, src_seq_len, first) 
        state_re = self.prepare_state(src_tokens_clone, src_seq_len_clone, first_clone)
        decoder_output_ner = self.decoder(tgt_tokens, state, update_tree) 
        decoder_output_re = self.decoder(re_tgt_tokens, state_re, update_tree)
        
        if isinstance(decoder_output_ner, torch.Tensor):
            return {'pred': decoder_output_ner, 're_pred': decoder_output_re}
        elif isinstance(decoder_output_ner, (tuple, list)):
            return {'pred': decoder_output_ner[0], 're_pred': decoder_output_re[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")
        
class BartSeq2SeqModel_RE_NER(nn.Module):
    def __init__(self, encoder, decoder, decoder_re, **kwargs):
        # 调用父类的构造方法
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_re = decoder_re  # 初始化 decoder_re

    @staticmethod  # 将 build_model 改成静态方法，不再需要 cls
    def build_model(bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False):
        # 加载BART预训练模型
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens)
        encoder = model.encoder
        decoder = model.decoder
        model_re = BartModel.from_pretrained(bart_model)
        model_re.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens)
        decoder_re = model_re.decoder

        # 处理特殊字符
        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符处理
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index) > 1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index >= num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        # 转换为 FBartEncoder 和 FBartDecoder（自定义的解码器）
        encoder = FBartEncoder(encoder)
        if decoder_type is None:
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
            decoder_re = FBartDecoder(decoder_re, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type == 'avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp)
            decoder_re = CaGFBartDecoder(decoder_re, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=True, use_encoder_mlp=use_encoder_mlp)
            decoder_re = CaGFBartDecoder(decoder_re, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=True, use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        # 直接返回模型实例，而不再使用 cls
        return BartSeq2SeqModel_RE_NER(encoder=encoder, decoder=decoder, decoder_re=decoder_re)
    
    def prepare_state(self, src_tokens, src_seq_len=None, first=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        return state
    
    def forward(self, src_tokens, tgt_tokens, re_tgt_tokens, src_seq_len, re_tgt_seq_len, tgt_seq_len, first, update_tree=False):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        src_tokens_clone,  src_seq_len_clone, first_clone = src_tokens.clone(), src_seq_len.clone(), first.clone()
        state = self.prepare_state(src_tokens, src_seq_len, first) 
        state_re = self.prepare_state(src_tokens_clone, src_seq_len_clone, first_clone)
        decoder_output_ner = self.decoder(tgt_tokens, state, update_tree) 
        decoder_output_re = self.decoder_re(re_tgt_tokens, state_re, update_tree)
        assert decoder_output_ner.shape[0] == decoder_output_re.shape[0]
        
        if isinstance(decoder_output_ner, torch.Tensor):
            return {'pred': decoder_output_ner, 're_pred': decoder_output_re}
        elif isinstance(decoder_output_ner, (tuple, list)):
            return {'pred': decoder_output_ner[0], 're_pred': decoder_output_re[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")
        
class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new