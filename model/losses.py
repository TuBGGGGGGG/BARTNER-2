
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from .utils import get_ent_tgt_tokens

class Seq2SeqLoss(LossBase):
    def __init__(self, max_type_id):
        super().__init__()
        self.max_type_id = max_type_id
    def get_loss_old(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)

        weight = pred.new_ones(pred.size(-1))
        weight[0:self.max_type_id+1] = 1.1
        pred= pred.reshape((pred.shape[0]*pred.shape[1], -1))
        tgt_tokens= tgt_tokens.reshape((tgt_tokens.shape[0]*tgt_tokens.shape[1]))
        
        loss = F.cross_entropy(target=tgt_tokens, input=pred, weight=weight)
        # with open("/disk1/wxl/Desktop/DeepKE/example/baseline/BARTNER/loss_log/E.json","a") as f:
        #     import torch
        #     torch.set_printoptions(profile="full")
        #     f.write(str(loss*10000)+"\n"+str(tgt_tokens)+"\n")
        return loss

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        ent_tgt_tokens, valid_pred_index = get_ent_tgt_tokens(tgt_tokens[:, 1:], self.max_type_id, pred.shape[1], pred.shape[2]) # 去掉第一个0才行
        assert ent_tgt_tokens.shape == pred.shape, f'{ent_tgt_tokens.shape},{pred.shape}'
        bsz, max_ent, vocab = ent_tgt_tokens.shape

        ent_tgt_tokens = ent_tgt_tokens.reshape(bsz*max_ent, vocab).index_select(dim=0,index=valid_pred_index)
        pred = pred.reshape(bsz*max_ent, vocab).index_select(dim=0,index=valid_pred_index)
        assert ent_tgt_tokens.shape == pred.shape, f'{ent_tgt_tokens.shape},{pred.shape}'
    
        loss = F.binary_cross_entropy(target=ent_tgt_tokens, input=pred)
        
        return loss

