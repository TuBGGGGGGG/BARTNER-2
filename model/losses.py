
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from .utils import get_ent_tgt_tokens

class Seq2SeqLoss(LossBase):
    def __init__(self, max_type_id):
        super().__init__()
        self.max_type_id = max_type_id
    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
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

    def get_loss_new(self, tgt_tokens, tgt_seq_len, pred):
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

class Seq2Seq_RE_NER_Loss(LossBase):
    def __init__(self, max_type_id):
        super().__init__()
        self.max_type_id = max_type_id
    
    def get_loss(self, tgt_tokens, tgt_seq_len, pred, re_tgt_tokens, re_tgt_seq_len, re_pred):
        assert pred.shape[0] == re_pred.shape[0]
        ner_loss = self.get_loss_step(tgt_tokens, tgt_seq_len, pred)
        re_loss = self.get_loss_step(re_tgt_tokens, re_tgt_seq_len, re_pred)
        return  0.5 * re_loss + 0.5 * ner_loss
    
    def get_loss_step(self, tgt_tokens, tgt_seq_len, pred):
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
        return loss
    
class Seq2SeqLoss_with_self_tgt(LossBase):
    def __init__(self, max_type_id):
        super().__init__()
        self.max_type_id = max_type_id
        
    def get_loss(self, tgt_tokens, tgt_seq_len, pred, self_tgt, self_tgt_seq_len, self_tgt_pred):

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
        
        loss1 = F.cross_entropy(target=tgt_tokens, input=pred, weight=weight)

        self_tgt_seq_len = self_tgt_seq_len - 1
        mask2 = seq_len_to_mask(self_tgt_seq_len, max_len=self_tgt.size(1) - 1).eq(0)
        self_tgt = self_tgt[:, 1:].masked_fill(mask2, -100)

        weight2 = self_tgt_pred.new_ones(self_tgt_pred.size(-1))
        weight2[0:self.max_type_id+1] = 1.1
        self_tgt_pred= self_tgt_pred.reshape((self_tgt_pred.shape[0]*self_tgt_pred.shape[1], -1))
        self_tgt= self_tgt.reshape((self_tgt.shape[0]*self_tgt.shape[1]))
        
        loss2 = F.cross_entropy(target=self_tgt, input=self_tgt_pred, weight=weight2)

        return loss1 + 0.001*loss2

