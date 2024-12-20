
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
import numpy as np
from itertools import chain

class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, target_type='bpe'):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels+2  # +2是由于有前面有两个特殊符号，sos和eos

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.em = 0
        self.total = 0
        self.target_type = target_type  # 如果是span的话，必须是偶数的span，否则是非法的
        print(f"\nmetric类型: [SPAN]  !\n")

    def evaluate(self, target_span, pred, tgt_tokens):
        # print("target_span: ",target_span,"\n")
        # print("pred: ",pred,"\n")
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            # print("ts: ",ts," ps: ",ps,"\n")
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            pairs = []
            cur_pair = []
            if len(ps):
                for j in ps:
                    if j<self.word_start_index:
                        if self.target_type == 'span':
                            if len(cur_pair)>0 and len(cur_pair)%2==0:
                                if all([cur_pair[i]<=cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                    pairs.append(tuple(cur_pair+[j]))
                                # TODO 好嚴格，可以一次獲得j類型對應的不止一個實體？
                        else:
                            if len(cur_pair) > 0:
                                # if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                #     pairs.append(tuple(cur_pair + [j])) # TODO 做关系抽取不要求递增id
                                pairs.append(tuple(cur_pair + [j])) 
                                # TODO 处理不连续实体，或许可以优化一下，丢弃掉不是递增的那部分id，而不是直接把这个cur_pair丢弃。
                                # TODO 处理关系的话
                        cur_pair = []
                    else:
                        cur_pair.append(j) # word直接加进去
            pred_spans.append(pairs.copy())

            tp, fn, fp = _compute_tp_fn_fp(pairs, ts)
            self.fn += fn
            self.tp += tp
            self.fp += fp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        print("正确预测个数：", self.tp," 错误预测个数：",self.fp, " 未被预测的正确实体个数：",self.fn)
        res['f'] = round(f, 4)*100
        res['rec'] = round(rec, 4)*100
        res['pre'] = round(pre, 4)*100
        res['em'] = round(self.em/self.total, 4)
        if reset:
            self.total = 0
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.em = 0
        return res
    
class Seq2SeqREMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels,  rel_type_start, target_type='bpe'):
        super(Seq2SeqREMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels+2  # +2是由于有前面有两个特殊符号，sos和eos

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.em = 0
        self.total = 0
        self.target_type = target_type  # 如果是span的话，必须是偶数的span，否则是非法的
        self.rel_type_start = rel_type_start
        print(f"\nmetric类型: [RE]  rel_type_start: {rel_type_start}!\n")

    def evaluate(self, target_span, pred, tgt_tokens):
        # print("target_span: ",target_span,"\n")
        # print("pred: ",pred,"\n")
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            assert len(ts) % 3 == 0
            ts = [list(chain(*(ts[i*3:i*3+3]))) for i in range(len(ts)//3)]
            ts2 = []
            for ti in range(len(ts)):
                ts2_ti = []
                for x in ts[ti]:
                    if x >= self.rel_type_start:
                        ts2_ti.append(x)
                #ts2_ti.sort()
                ts2.append(ts2_ti)
            ts = ts2
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            pairs = []
            cur_pair = []
            if len(ps):
                for j in ps:
                    if j < self.word_start_index and j >= self.rel_type_start:
                        if len(cur_pair) > 0:
                            cur_pair = cur_pair + [j]
                            # cur_pair.sort()
                            pairs.append(tuple(cur_pair)) 
                        cur_pair = []
                    elif j < self.rel_type_start:
                        continue
                    else:
                        cur_pair.append(j) # word直接加进去
            pred_spans.append(pairs.copy())

            # print("pairs: ",pairs, " ts: ",ts,"\n")
            tp, fn, fp = _compute_tp_fn_fp(pairs, ts)
            self.fn += fn
            self.tp += tp
            self.fp += fp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        print("正确预测个数：", self.tp," 错误预测个数：",self.fp, " 未被预测的正确实体个数：",self.fn)
        res['f'] = round(f, 4)*100
        res['rec'] = round(rec, 4)*100
        res['pre'] = round(pre, 4)*100
        res['em'] = round(self.em/self.total, 4)
        if reset:
            self.total = 0
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.em = 0
        return res    
    
class Seq2Seq_RE_NER_Metric(MetricBase):
    def __init__(self, eos_token_id, num_labels,  rel_type_start, target_type='bpe'):
        super(Seq2Seq_RE_NER_Metric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels+2  # +2是由于有前面有两个特殊符号，sos和eos

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.em = 0
        self.total = 0

        self.fp_re = 0
        self.tp_re = 0
        self.fn_re = 0
        self.em_re = 0
        self.total_re = 0

        self.target_type = target_type  # 如果是span的话，必须是偶数的span，否则是非法的
        self.rel_type_start = rel_type_start
        print(f"\nmetric类型: [NER & RE]  rel_type_start: {rel_type_start}!\n")

    def evaluate(self, target_span, pred, tgt_tokens, re_target_span, re_pred, re_tgt_tokens):

        self.evaluate_ner(target_span, pred, tgt_tokens)
        self.evaluate_re(re_target_span, re_pred, re_tgt_tokens)

    def evaluate_ner(self, target_span, pred, tgt_tokens):

        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            pairs = []
            cur_pair = []
            if len(ps):
                for j in ps:
                    if j<self.word_start_index:
                        if self.target_type == 'span':
                            if len(cur_pair)>0 and len(cur_pair)%2==0:
                                if all([cur_pair[i]<=cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                    pairs.append(tuple(cur_pair+[j]))
                        else:
                            if len(cur_pair) > 0:
                                if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                    pairs.append(tuple(cur_pair + [j])) 
                        cur_pair = []
                    else:
                        cur_pair.append(j) # word直接加进去
            pred_spans.append(pairs.copy())

            tp, fn, fp = _compute_tp_fn_fp(pairs, ts)
            self.fn += fn
            self.tp += tp
            self.fp += fp

    def evaluate_re(self, target_span_re, pred_re, tgt_tokens_re):
        
        self.total_re += pred_re.size(0)
        pred_eos_index = pred_re.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens_re.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred_re = pred_re[:, 1:]  # 去掉</s>
        tgt_tokens_re = tgt_tokens_re[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span_re, pred_re.tolist())):
            assert len(ts) % 3 == 0
            ts = [list(chain(*(ts[i*3:i*3+3]))) for i in range(len(ts)//3)]
            ts2 = []
            for ti in range(len(ts)):
                ts2_ti = []
                for x in ts[ti]:
                    if x >= self.rel_type_start:
                        ts2_ti.append(x)
                ts2.append(ts2_ti)
            ts = ts2
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens_re[i, :target_seq_len[i]].eq(pred_re[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em_re += em
            pairs = []
            cur_pair = []
            if len(ps):
                for j in ps:
                    if j < self.word_start_index and j >= self.rel_type_start:
                        if len(cur_pair) > 0:
                            cur_pair = cur_pair + [j]
                            pairs.append(tuple(cur_pair)) 
                        cur_pair = []
                    elif j < self.rel_type_start:
                        continue
                    else:
                        cur_pair.append(j) # word直接加进去
            pred_spans.append(pairs.copy())

            tp, fn, fp = _compute_tp_fn_fp(pairs, ts)
            self.fn_re += fn
            self.tp_re += tp
            self.fp_re += fp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        print("[NER]: 正确预测个数：", self.tp," 错误预测个数：",self.fp, " 未被预测的正确实体个数：",self.fn)
        res['f'] = round(f, 4)*100
        res['rec'] = round(rec, 4)*100
        res['pre'] = round(pre, 4)*100
        res['em'] = round(self.em/self.total, 4)

        f_re, pre_re, rec_re = _compute_f_pre_rec(1, self.tp_re, self.fn_re, self.fp_re)
        print("[RE]: 正确预测个数：", self.tp_re," 错误预测个数：",self.fp_re, " 未被预测的正确实体个数：",self.fn_re)
        res['f_RE'] = round(f_re, 4)*100
        res['rec_RE'] = round(rec_re, 4)*100
        res['pre_RE'] = round(pre_re, 4)*100
        res['em_RE'] = round(self.em_re/self.total_re, 4)

        if reset:
            self.total = 0
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.em = 0

            self.total_re = 0
            self.fp_re = 0
            self.tp_re = 0
            self.fn_re = 0
            self.em_re = 0
        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, (set, list, np.ndarray)):
        ts = {tuple(key):1 for key in list(ts)}
    if isinstance(ps, (set, list, np.ndarray)):
        ps = {tuple(key):1 for key in list(ps)}

    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp
