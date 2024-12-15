from torch import nn
import torch


class MultiHeadBiaffine(nn.Module):
    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim%n_head==0
        in_head_dim = dim//n_head
        out = dim if out is None else out
        assert out%n_head == 0
        out_head_dim = out//n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """
        :param h: bsz x max_ent x dim
        :param v: vocabsize x dim
        :return: bsz x max_ent x vocabsize
        """
        bsz, max_ent, dim = h.size()  # bsz x max_ent x dim
        # Expand v to match batch size
        if len(v.shape) == 2:
            vocabsize, dim = v.size()  
            v = v.unsqueeze(0).expand(bsz, -1, -1)  # bsz x vocabsize x dim, tag_scores, eos_scores.
        else:
            bsz_, vocabsize, dim = v.size()  # word scores
            assert bsz == bsz_

        # Reshape for multi-head computation
        h = h.reshape(bsz, max_ent, self.n_head, -1)  # bsz x max_ent x n_head x in_head_dim
        v = v.reshape(bsz, vocabsize, self.n_head, -1)  # bsz x vocabsize x n_head x in_head_dim

        # Biaffine calculation using einsum
        w = torch.einsum('bmhx,hdxy,bnhy->bhdmn', h, self.W, v)  # bsz x n_head x out_head_dim x max_ent x vocabsize
        w = w.reshape(bsz, self.out_dim, max_ent, vocabsize)  # bsz x out_dim x max_ent x vocabsize

        # If out_dim = 1, squeeze the extra dimension
        if self.out_dim == 1:
            w = w.squeeze(1)  # bsz x max_ent x vocabsize
        return w