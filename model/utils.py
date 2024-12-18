
import numpy as np
import fitlog
from fastNLP import SortedSampler, Tester
from .metrics import Seq2SeqSpanMetric
from collections import defaultdict
import torch
from fastNLP import seq_len_to_mask
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP import SequentialSampler
from fastNLP import DataSetIter,DataSet
import torch.nn.functional as F
import time
import json
from copy import deepcopy


def get_max_len_max_len_a(data_bundle, max_len=10):
    """
    当给定max_len=10的时候计算一个最佳的max_len_a

    :param data_bundle:
    :param max_len:
    :return:
    """
    max_len_a = -1
    for name, ds in data_bundle.iter_datasets():
        if name=='train':continue
        src_seq_len = np.array(ds.get_field('src_seq_len').content)
        tgt_seq_len = np.array(ds.get_field('tgt_seq_len').content)
        _len_a = round(max(np.maximum(tgt_seq_len - max_len+2, 0)/src_seq_len), 1)

        if _len_a>max_len_a:
            max_len_a = _len_a

    return max_len, max_len_a

def current_tree_init(ds, pipe):
    num = 0
    num2 = 0
    num_x = 0
    num_zero_ent = 0
    ds_ = DataSet()

    current_tree = {}

    is_ordered_key = set()
    split_list = ["entities","entity_tags","entity_spans","tgt_tokens","target_span","tgt_seq_len"]
    
    for id, ds_i in enumerate(ds):
        before = {}

        for i in split_list:
            before[i]=ds_i[i]

        num2 += len(before["entity_tags"])
        key = ' '.join(ds_i["raw_words"])
        if key in current_tree.keys():
            continue # 重复数据
        value = {
            "ds_indexes": [],
            "position": 1,
            "stop_position": ds_i["tgt_seq_len"]
        }
        '''
            [position]: 未排序的第一个位置
            [stop_position]: 排序结束的位置，初始化为key对应的原始tgt_seq_len
        '''
        
        num+=len(before["entity_tags"])
        if len(before["entity_tags"]) <=1:
            num_x += 1 

        for i in range(len(before["entity_tags"])):
            split_i = {
                "entities":before["entities"][i:i+1],
                "entity_tags":before["entity_tags"][i:i+1],
                "entity_spans":before["entity_spans"][i:i+1],
                "target_span":before["target_span"][i:i+1]
            }
            split_i["tgt_tokens"] = [0] + split_i["target_span"][0] + [1]
            split_i["tgt_seq_len"] = len(split_i["tgt_tokens"])

            x = deepcopy(ds_i)
            for j in split_list:
                x[j] = split_i[j]

            ds_.append(x)

            value["ds_indexes"].append(len(ds_)-1)

        if len(value["ds_indexes"]) <= 1:
            value["position"] = value["stop_position"] # 0个或1个实体，不用排序
            is_ordered_key.add(key)
            if len(value["ds_indexes"]) == 0:
                num_zero_ent += 1
                x = deepcopy(ds_i)
                ds_.append(x) # 0个实体，直接将原数据加回去
                value["ds_indexes"].append(len(ds_)-1)
        current_tree[key] = value # 处理完之后加入字典树
    
    ds_.set_ignore_type('target_span', 'entities')
    ds_.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    print("非去重实体个数",num2," 去重实体个数： ",num," 新训练数据个数：",len(ds_)," 不重复数据个数: ", len(current_tree), " 0实体数据个数：",num_zero_ent)
    print(num_x)
    assert len(ds_) == num+num_zero_ent
    return ds_, current_tree, is_ordered_key

@torch.no_grad()
def get_update_loss(tgt_tokens, tgt_seq_len, pred):
    """
    :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
    :param pred: bsz x max_len-1 x vocab_size
    :return:
    """
    tgt_seq_len = tgt_seq_len - 1
    mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
    tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
    loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1,2), reduction='none').sum(-1) / tgt_seq_len
    return loss

@torch.no_grad()
def get_loss_all(ds, model):
    # 下标或者add/delete操作会变化
    sampler = SequentialSampler()
    batch = DataSetIter(batch_size=1024, dataset=ds, sampler=sampler)
    model.eval()
    loss_all = torch.empty(0, device='cuda')
    for id,i in enumerate(batch):
        i[0]["update_tree"]=True
        _move_dict_value_to_device(i[0],i[-1],device=torch.device('cuda'))
        tgt_tokens = i[0]['tgt_tokens']
        tgt_seq_len = i[-1]['tgt_seq_len']
        
        pred = model(**i[0])['pred']
        
        batch_loss= get_update_loss(tgt_tokens, tgt_seq_len, pred)
        
        loss_all = torch.cat((loss_all, batch_loss))
    return loss_all

@torch.no_grad()
def get_span_all(ds, model):
    # 下标或者add/delete操作会变化
    sampler = SequentialSampler()
    batch = DataSetIter(batch_size=1024, dataset=ds, sampler=sampler)
    model.eval()
    loss_all = torch.empty(0, device='cuda')
    for id,i in enumerate(batch):
        i[0]["update_tree"]=True
        _move_dict_value_to_device(i[0],i[-1],device=torch.device('cuda'))
        tgt_tokens = i[0]['tgt_tokens']
        tgt_seq_len = i[-1]['tgt_seq_len']
        
        pred = model(**i[0])['pred']
        
        batch_loss= get_update_loss(tgt_tokens, tgt_seq_len, pred)
        
        loss_all = torch.cat((loss_all, batch_loss))
    return loss_all

@torch.no_grad()
def update_func(current_tree1, is_ordered_key1, ds_,loss_all, pipe):
    # ds_是分化后的训练数据集
    current_tree = deepcopy(current_tree1)
    assert current_tree[list(current_tree.keys())[-1]]["ds_indexes"][-1] == len(ds_) - 1,f'{current_tree[list(current_tree.keys())[-1]]["ds_indexes"][-1]} {len(ds_) - 1}'
    is_ordered_key = deepcopy(is_ordered_key1)
    ds2 = DataSet()
    split_list = ["entities","entity_tags","entity_spans","tgt_tokens","target_span","tgt_seq_len"]
    add_list = ["entities","entity_tags","entity_spans","target_span"] # 直接将最后一个加在hard尾巴的key
    for key in current_tree.keys():
        if len(current_tree[key]["ds_indexes"]) == 1 or current_tree[key]['position'] == current_tree[key]['stop_position']:
            assert key in is_ordered_key,f'{key},{current_tree[key],{ds_[0]}}'
        if key in is_ordered_key:
            # 已经排序完毕的key
            assert len(current_tree[key]['ds_indexes']) == 1 and current_tree[key]['position'] == current_tree[key]['stop_position']

            ds2.append(ds_[current_tree[key]['ds_indexes'][0]])
            current_tree[key]["ds_indexes"] = [len(ds2)-1] # 更新一下数据索引
            continue

        position = current_tree[key]['position']
        ds_indexes = current_tree[key]['ds_indexes']
        loss_unordered = loss_all[ds_indexes]
        hard_node_id = ds_indexes[loss_unordered.argmax()] # 困惑度最大的作为下一个。
        # hard_node_id = ds_indexes[0]

        # tgt_tokens_len_list = torch.tensor([ds_[i]['tgt_seq_len'] for i in ds_indexes])
        # hard_node_id = ds_indexes[tgt_tokens_len_list.argmax()] # span最长的作为下一个。
        
        ds_hard = ds_[hard_node_id]
        
        hard_value = {}
        for i in split_list:
            hard_value[i] = ds_hard[i]
        hard_value["tgt_tokens"] = hard_value["tgt_tokens"][:-1]

        new_ds_indexes=[]

        for id in ds_indexes:
            if id==hard_node_id:
                continue

            add_value = {}
            for i in add_list:
                add_value[i] = ds_[id][i][-1:]

            add_value["tgt_tokens"] = ds_[id]["tgt_tokens"][position:]

            add_ds = deepcopy(ds_hard)
            for i in add_list:
                add_ds[i] = hard_value[i] + add_value[i]
            add_ds["tgt_tokens"] = hard_value["tgt_tokens"] + add_value["tgt_tokens"]
            add_ds["tgt_seq_len"] = len(add_ds["tgt_tokens"])

            ds2.append(add_ds)

            new_ds_indexes.append(len(ds2)-1)
        
        current_tree[key]["ds_indexes"] = new_ds_indexes
        if len(new_ds_indexes) == 1:
            new_position = len(ds2[-1]["tgt_tokens"]) # 理论上=stop_position
            assert new_position == current_tree[key]["stop_position"], f'{new_position} {current_tree[key]["stop_position"]} {ds2[-1]["tgt_tokens"]}, {len(ds2)}'
        else:
            new_position = len(hard_value["tgt_tokens"])

        current_tree[key]["position"] = new_position
        if new_position == current_tree[key]["stop_position"]:
            assert len(new_ds_indexes)==1,f'分化数据归一才能算排序完成！！！'
            is_ordered_key.add(key)
        ds2.set_ignore_type('target_span', 'entities')
        ds2.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        ds2.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

        ds2.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        ds2.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        ds2.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
        ds2.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    return current_tree, is_ordered_key, ds2

@torch.no_grad()
def update_func_span(current_tree1, is_ordered_key1, ds_, false_weight_tensor, pipe):
    # ds_是分化后的训练数据集
    current_tree = deepcopy(current_tree1)
    assert current_tree[list(current_tree.keys())[-1]]["ds_indexes"][-1] == len(ds_) - 1,f'{current_tree[list(current_tree.keys())[-1]]["ds_indexes"][-1]} {len(ds_) - 1}'
    is_ordered_key = deepcopy(is_ordered_key1)
    ds2 = DataSet()
    split_list = ["entities","entity_tags","entity_spans","tgt_tokens","target_span","tgt_seq_len"]
    add_list = ["entities","entity_tags","entity_spans","target_span"] # 直接将最后一个加在hard尾巴的key
    for key in current_tree.keys():
        if len(current_tree[key]["ds_indexes"]) == 1 or current_tree[key]['position'] == current_tree[key]['stop_position']:
            assert key in is_ordered_key,f'{key},{current_tree[key],{ds_[0]}}'
        if key in is_ordered_key:
            # 已经排序完毕的key
            assert len(current_tree[key]['ds_indexes']) == 1 and current_tree[key]['position'] == current_tree[key]['stop_position']

            ds2.append(ds_[current_tree[key]['ds_indexes'][0]])
            current_tree[key]["ds_indexes"] = [len(ds2)-1] # 更新一下数据索引
            continue

        position = current_tree[key]['position']
        ds_indexes = current_tree[key]['ds_indexes']
        false_weight = false_weight_tensor[ds_indexes]
        hard_node_id = ds_indexes[false_weight.argmax()]

        # tgt_tokens_len_list = torch.tensor([ds_[i]['tgt_seq_len'] for i in ds_indexes])
        # hard_node_id = ds_indexes[tgt_tokens_len_list.argmax()]
        
        ds_hard = ds_[hard_node_id]
        
        hard_value = {}
        for i in split_list:
            hard_value[i] = ds_hard[i]
        hard_value["tgt_tokens"] = hard_value["tgt_tokens"][:-1]

        new_ds_indexes=[]

        for id in ds_indexes:
            if id==hard_node_id:
                continue

            add_value = {}
            for i in add_list:
                add_value[i] = ds_[id][i][-1:]

            add_value["tgt_tokens"] = ds_[id]["tgt_tokens"][position:]

            add_ds = deepcopy(ds_hard)
            for i in add_list:
                add_ds[i] = hard_value[i] + add_value[i]
            add_ds["tgt_tokens"] = hard_value["tgt_tokens"] + add_value["tgt_tokens"]
            add_ds["tgt_seq_len"] = len(add_ds["tgt_tokens"])

            ds2.append(add_ds)

            new_ds_indexes.append(len(ds2)-1)
        
        current_tree[key]["ds_indexes"] = new_ds_indexes
        if len(new_ds_indexes) == 1:
            new_position = len(ds2[-1]["tgt_tokens"]) # 理论上=stop_position
            assert new_position == current_tree[key]["stop_position"], f'{new_position} {current_tree[key]["stop_position"]} {ds2[-1]["tgt_tokens"]}, {len(ds2)}'
        else:
            new_position = len(hard_value["tgt_tokens"])

        current_tree[key]["position"] = new_position
        if new_position == current_tree[key]["stop_position"]:
            assert len(new_ds_indexes)==1,f'分化数据归一才能算排序完成！！！'
            is_ordered_key.add(key)
        ds2.set_ignore_type('target_span', 'entities')
        ds2.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        ds2.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

        ds2.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        ds2.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        ds2.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
        ds2.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    return current_tree, is_ordered_key, ds2

@torch.no_grad()
def update_tree(ds, model, current_tree, is_ordered_key, pipe, mode, max_type_id):
    # 通过get_loss_all获得在训练集上的每一条loss
    if mode == "loss":
        # 困惑度筛选
        loss_all = get_loss_all(ds, model)
        # 再通过update_func获得新的ds,current_tree, is_ordered_key
        new_current_tree, new_is_ordered_key, new_ds = update_func(current_tree, is_ordered_key, ds, loss_all, pipe)
    elif mode == "false_span":
        loss_all = get_loss_all(ds, model)
        false_weight_tensor = get_update_pred_merge(model,ds,max_type_id, loss_all)
        new_current_tree, new_is_ordered_key, new_ds = update_func_span(current_tree, is_ordered_key, ds, false_weight_tensor, pipe)
    return new_current_tree, new_is_ordered_key, new_ds

@torch.no_grad()
def get_final_dataset(ds_, ds_old, pipe, current_tree):
    # ds_new = deepcopy(ds_old)
    ds_new = DataSet()
    for i in ds_old:
        key = ' '.join(i["raw_words"])
        id = current_tree[key]["ds_indexes"][-1]
        add_data = deepcopy(ds_[id])
        ds_new.append(add_data)

    ds_new.set_ignore_type('target_span', 'entities')
    ds_new.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_new.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_new.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_new.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_new.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_new.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    # ds_new.save("/disk1/wxl/Desktop/DeepKE/example/baseline/BARTNER/caches/ds_final_11_19_569.pt")

    for i in ds_old:
        key = ' '.join(i["raw_words"])
        id = current_tree[key]["ds_indexes"][-1]
        add_data = deepcopy(ds_[id])
        ds_new.append(add_data)

    ds_new.set_ignore_type('target_span', 'entities')
    ds_new.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_new.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_new.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_new.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_new.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_new.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    return ds_new

@torch.no_grad()
def get_tmp_dataset(ds_, ds_old, pipe, current_tree):
    ds_new = DataSet()
    for i in ds_old:
        key = ' '.join(i["raw_words"])
        id = current_tree[key]["ds_indexes"]
        for j in id:
            add_data = deepcopy(ds_[j]) # deepcopy应该不影响，因为ds_new的数据每一轮用完就更新了，并没有进行更改。
            ds_new.append(add_data)
    ds_new.set_ignore_type('target_span', 'entities')
    ds_new.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_new.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_new.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_new.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_new.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_new.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    return ds_new

@torch.no_grad()
def all_key_set_init(ds):
    all_update_key = set()
    for i in ds:
        key = ' '.join(i["raw_words"])
        all_update_key.add(key)
    return all_update_key

@torch.no_grad()
def get_span_from_pred(pred, max_type_id):
    cur_span = []
    all_spans = []
    for i in pred[1:].item():
        if i == 1:
            break
        if i > max_type_id:
            cur_span.append(i)
        else:
            cur_span.append(i)
            all_spans.append(str(cur_span)) # str是为了可hash
            cur_span = []
    return all_spans

@torch.no_grad()
def pad_spans(x, max_len):
    pad_value = x[-1]  # 填充值
    num_new_rows = max_len - len(x)  # 要添加的行数
    pad_array = np.full((num_new_rows, 2), pad_value)
    result = np.vstack((x, pad_array))
    return result

@torch.no_grad()
def get_ent_boundary_from_tgt_token(tgt_token, min_tag_id):
    assert tgt_token[0]==0
    cur_span = []
    all_spans = []
    l = 0
    for id,i in enumerate(tgt_token):
        # assert i != 1,tgt_token
        if i == -1:
            assert cur_span == [],f"{tgt_token},{cur_span},{all_spans}"
            break # eval模式，-1为pad，不参与任何实体
        if i <= 2:
            cur_span = [l, id+1]
            l = id+1
            all_spans.append(cur_span)
            cur_span = []
            if i == 2:
                break # 保证了最后一个ent是2,eos
        elif i >= min_tag_id:
            cur_span = [l, id+1]
            l = id+1
            all_spans.append(cur_span)
            cur_span = []
    return all_spans,len(all_spans)

@torch.no_grad()
def get_ent_boundary(tgt_tokens, min_tag_id):
    # print("不准调用！")
    all_spans = []
    all_spans_len = []
    max_len_span = 0
    for id,i in enumerate(tgt_tokens):
        spans, len_spans = get_ent_boundary_from_tgt_token(i, min_tag_id)
        all_spans_len.append(len_spans)
        max_len_span = max(max_len_span, len_spans)
        all_spans.append(spans)
    
    all_spans = [np.array(i) for i in all_spans]
    all_spans = [pad_spans(i, max_len_span) for i in all_spans]
    all_spans = torch.from_numpy(np.array(all_spans)).to('cuda')
    ent_decode_mask = torch.zeros(all_spans.shape[0:2], device=torch.device('cuda'))
    for id,i in enumerate(all_spans_len):
        ent_decode_mask[id][i:] = 1
    return all_spans, ent_decode_mask

def get_ent_tgt_token(tgt_token, max_type_id, max_ent, max_vocab):
    assert tgt_token[0]!=0 and (tgt_token[-1] == 1),tgt_token
    seqs = torch.zeros((max_ent, max_vocab),  device=torch.device('cuda'), requires_grad=False)
    cur_span = []
    ent_id = 0
    for id,i in enumerate(tgt_token):
        cur_span.append(i)
        if i <= max_type_id:
            cur_span = torch.tensor(cur_span)
            seqs[ent_id][cur_span] = 1
            ent_id += 1
            cur_span = []
            if i == 1:
                seqs[ent_id:,1] = 1 # pad_ent的预测target应该全为eos实体，也就是1
                break # 保证了最后一个ent是1,eos
    assert cur_span == [],tgt_token
    return seqs.unsqueeze(0), ent_id

def get_ent_tgt_tokens(tgt_tokens, max_type_id, max_ent, max_vocab):
    # print("不准调用！")
    all_seqs = []
    valid_pred_index = []
    for id,i in enumerate(tgt_tokens):
        seq, ent_id = get_ent_tgt_token(i, max_type_id, max_ent, max_vocab)
        valid_pred_index += range(id*max_ent, id*max_ent+ent_id)
        all_seqs.append(seq)
    
    all_seqs = torch.cat(all_seqs).to('cuda')
    all_seqs = all_seqs
    all_seqs.requires_grad = False
    valid_pred_index = torch.tensor(valid_pred_index, device=torch.device('cuda'))
    return all_seqs, valid_pred_index

@torch.no_grad()
def update_ds(model, ds, is_ordered_key, max_type_id, pipe):
    model.eval()
    ds_ = DataSet()
    sampler = SequentialSampler()
    batch = DataSetIter(batch_size=1, dataset=ds, sampler=sampler)
    assert len(ds)==len(batch)
    for id, i in enumerate(batch):
        ds_i = deepcopy(ds[id])
        key = ' '.join(ds_i["raw_words"])
        if key in is_ordered_key:
            ds_.append(ds_i)
            continue
        src_tokens=i[0]["src_tokens"].to('cuda')
        src_seq_len=i[0]["src_seq_len"].to('cuda')
        first=i[0]["first"].to('cuda')
        pred = model.predict(src_tokens.to('cuda'), src_seq_len.to('cuda'), first.to('cuda'))['pred']

        target_spans = [str(span) for span in ds_i["target_span"]]
        pred_spans = get_span_from_pred(pred, max_type_id)
        target_spans, pred_spans = set(target_spans), set(pred_spans)
        false_spans = target_spans - pred_spans # 没预测出来的span
        true_spans = target_spans - false_spans # 成功预测的span
        if len(false_spans)==0:
            is_ordered_key.add(key)
            ds_.append(ds_i)
            continue 
        # 重组ds_i
        split_list = ["entities","entity_tags","entity_spans","tgt_tokens","target_span","tgt_seq_len"]
        add_list = ["entities","entity_tags","entity_spans","target_span"] # 直接将最后一个加在hard尾巴的key
        add_value = {"entities": [], "entity_tags": [], "entity_spans": [], "target_span": [], "tgt_tokens": [0]}
        for j in false_spans:
            index = ds_i["target_span"].index(json.loads(j))
            for k in add_list:
                add_value[k].append(ds_i[k][index])
            add_value["tgt_tokens"] += ds_i["target_span"][index]

        for j in true_spans:
            index = ds_i["target_span"].index(json.loads(j))
            for k in add_list:
                add_value[k].append(ds_i[k][index])
            add_value["tgt_tokens"] += ds_i["target_span"][index]

        for k in add_list:
            ds_i[k] = add_value[k]
        xxx = deepcopy(ds_i["tgt_tokens"])
        ds_i["tgt_tokens"] = add_value["tgt_tokens"] + [1]
        assert len(ds_i["tgt_tokens"]) == ds_i["tgt_seq_len"], f"{len(ds_i['tgt_tokens'])},{ds_i['tgt_seq_len']},{ds_i['tgt_tokens']},{xxx}"
        ds_.append(ds_i)
    
    model.train()
    ds_.set_ignore_type('target_span', 'entities')
    ds_.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    return ds_, is_ordered_key

@torch.no_grad()
def all_key_set_init(ds):
    all_update_key = set()
    for i in ds:
        key = ' '.join(i["raw_words"])
        all_update_key.add(key)
    return all_update_key

@torch.no_grad()
def get_span_from_pred(pred, max_type_id):
    assert len(pred.shape)==1,pred
    cur_span = []
    all_spans = []
    pred = pred.tolist()
    for i in pred[1:]:
        if i == 1:
            break
        if i > max_type_id:
            cur_span.append(i)
        else:
            cur_span.append(i)
            all_spans.append(str(cur_span)) # str是为了可hash
            cur_span = []
    return all_spans

@torch.no_grad()
def get_RE_from_pred(pred, max_type_id, rel_start_id):
    assert len(pred.shape)==1,pred
    cur_span = []
    all_spans = []
    pred = pred.tolist()
    for i in pred[1:]:
        if i == 1:
            break
        if i > max_type_id or i < rel_start_id:
            cur_span.append(i)
        else:
            cur_span.append(i)
            cur_span.sort()
            all_spans.append(str(cur_span)) # str是为了可hash
            cur_span = []
    return all_spans

@torch.no_grad()
def get_update_pred(model, ds, max_type_id):
    sampler = SequentialSampler()
    batch = DataSetIter(batch_size=48, dataset=ds, sampler=sampler)
    false_spans_list = []
    true_spans_list = []
    ds_index = 0
    true_num = 0
    false_num = 0
    for id, i in enumerate(batch):
        src_tokens=i[0]["src_tokens"].to('cuda')
        src_seq_len=i[0]["src_seq_len"].to('cuda')
        first=i[0]["first"].to('cuda')
        pred = model.predict(src_tokens, src_seq_len, first)['pred'] # batchsize x maxlen
        assert len(pred.shape)==2,pred.shape
        for j in pred:
            target_spans = [str(span) for span in ds[ds_index]["target_span"]]
            ds_index += 1
            pred_spans = get_span_from_pred(j, max_type_id)
            target_spans, pred_spans = set(target_spans), set(pred_spans)
            false_spans = target_spans - pred_spans # 没预测出来的span
            true_spans = target_spans - false_spans # 成功预测的span
            true_num += len(true_spans)
            false_num += len(false_spans)

            false_spans_list.append(false_spans)
            true_spans_list.append(true_spans)
    return false_spans_list, true_spans_list

@torch.no_grad()
def get_update_pred_merge(model, ds, max_type_id, loss_all):
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    sampler = SequentialSampler()
    batch = DataSetIter(batch_size=48, dataset=ds, sampler=sampler)
    false_spans_list = []
    ds_index = 0
    for id, i in enumerate(batch):
        src_tokens=i[0]["src_tokens"].to('cuda')
        src_seq_len=i[0]["src_seq_len"].to('cuda')
        first=i[0]["first"].to('cuda')
        pred = model.predict(src_tokens, src_seq_len, first)['pred'] # batchsize x maxlen
        assert len(pred.shape)==2,pred.shape
        for j in pred:
            target_spans = [str(span) for span in ds[ds_index]["target_span"]]
            pred_spans = get_span_from_pred(j, max_type_id)
            
            target_spans, pred_spans = set(target_spans), set(pred_spans)
            
            false_spans = target_spans - pred_spans # 没预测出来的span
            false_spans_list.append(len(false_spans)+sigmoid(loss_all[ds_index]))
            ds_index += 1
    return torch.tensor(false_spans_list, device=torch.device('cuda'))
    
@torch.no_grad()
def get_reorder_ds(ds, is_ordered_key, pipe, false_spans_list, true_spans_list):
    ds_ = DataSet()
    for id, (false_spans, true_spans) in enumerate(zip(false_spans_list, true_spans_list)):
        ds_i = deepcopy(ds[id])
        key = ' '.join(ds_i["raw_words"])
        if key in is_ordered_key:
            ds_.append(ds_i)
            continue
        if len(false_spans)==0:
            is_ordered_key.add(key)
            ds_.append(ds_i)
            continue 
        # 重组ds_i
        add_list = ["entities","entity_tags","entity_spans","target_span"] # 直接将最后一个加在hard尾巴的key
        add_value = {"entities": [], "entity_tags": [], "entity_spans": [], "target_span": [], "tgt_tokens": [0]}

        for j in true_spans:
            index = ds_i["target_span"].index(json.loads(j))
            for k in add_list:
                add_value[k].append(ds_i[k][index])
            add_value["tgt_tokens"] += ds_i["target_span"][index]
            
        for j in false_spans:
            try:
                index = ds_i["target_span"].index(json.loads(j))
            except:
                print(ds_i["target_span"])
                print(json.loads(j))
                exit()
            for k in add_list:
                add_value[k].append(ds_i[k][index])
            add_value["tgt_tokens"] += ds_i["target_span"][index]


        for k in add_list:
            ds_i[k] = add_value[k]
        xxx = deepcopy(ds_i["tgt_tokens"])
        ds_i["tgt_tokens"] = add_value["tgt_tokens"] + [1]
        assert len(ds_i["tgt_tokens"]) == ds_i["tgt_seq_len"], f"{len(ds_i['tgt_tokens'])},{ds_i['tgt_seq_len']},{ds_i['tgt_tokens']},{xxx}"
        ds_.append(ds_i)
    
    ds_.set_ignore_type('target_span', 'entities')
    ds_.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    return ds_, is_ordered_key

@torch.no_grad()
def update_ds_batch(model, ds, is_ordered_key, max_type_id, pipe):
    model.eval()
    time1 = time.time()
    false_spans_list, true_spans_list = get_update_pred(model, ds, max_type_id)
    print("函数get_update_pred用时：",time.time()-time1)
    assert len(ds)==len(true_spans_list)==len(false_spans_list)
    time1 = time.time()
    ds, is_ordered_key = get_reorder_ds(ds, is_ordered_key, pipe, false_spans_list, true_spans_list)
    print("函数get_reorder_ds用时：",time.time()-time1)
    model.train()
    return ds, is_ordered_key

@torch.no_grad()
def get_double_ds(ds, pipe):
    ds_new = deepcopy(ds)
    for i in ds:
        add_data = deepcopy(i)
        ds_new.append(add_data)

    ds_new.set_ignore_type('target_span', 'entities')
    ds_new.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_new.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_new.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_new.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_new.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_new.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    print("double_ds done!")
    return ds_new

@torch.no_grad()
def get_ds(ds, pipe):
    ds_new = deepcopy(ds)
    for i in ds:
        add_data = deepcopy(i)
        ds_new.append(add_data)

    ds_new.set_ignore_type('target_span', 'entities')
    ds_new.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
    ds_new.set_pad_val('src_tokens', pipe.tokenizer.pad_token_id)

    ds_new.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
    ds_new.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
    ds_new.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
    ds_new.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')
    ds_new.save("/disk1/wxl/Desktop/DeepKE/example/baseline/BARTNER/caches/ds_final_longent.pt")
    return ds


