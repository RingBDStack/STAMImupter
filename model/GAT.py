import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import random
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)
        self.outdim = fea[self.L]

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x = inputs[:, :, :, :self.outdim] + x
            x = self.ln(x)
        return x


class Dynamic_G_with_Attention(nn.Module):
    def __init__(self, static_adj, sample, heads, attn_ratio=0.5, add_attn_adp=True, adp_ratio=0.2):
        super().__init__()
        adj = np.zeros((static_adj.shape[0], static_adj.shape[0]), dtype=int)
        for i in range(adj.shape[0]):
            adj[i, static_adj[i]] = 1
        self.sa = torch.from_numpy(adj).to("cuda:0")
        self.h = heads
        self.attnr = attn_ratio
        self.adpr = adp_ratio
        self.addadp = add_attn_adp
        self.num_nodes = static_adj.shape[0]
        if add_attn_adp:
            self.adpvec = nn.Parameter(torch.randn(self.num_nodes, sample + 2 * (int(sample / 2))))
            # self.adpvec = nn.Parameter(torch.randn(self.num_nodes, sample))

    def forward(self, attn_sample, cp, formal_rate=0.4, dyna_adj=None):
        B, T, S, N = attn_sample.shape
        attn = attn_sample.unsqueeze(-3).expand(B, T, N, S, N)[
               torch.arange(B)[:, None, None, None],
               torch.arange(T)[None, :, None, None],
               torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)

        attn_sumT = self.attnr * attn[:, -1, :, :] + (1 - self.attnr) * torch.mean(attn[:, :-1, :, :], dim=1)
        batch_sa = self.sa.unsqueeze(0).expand(B, N, N)
        res = attn_sumT * batch_sa
        if self.addadp:
            attns_sumT = self.attnr * attn_sample[:, -1, :, :] + (1 - self.attnr) * torch.mean(
                attn_sample[:, :-1, :, :], dim=1)
            batch_adpvec = self.adpvec.unsqueeze(0).expand(B, N, S)
            adp_a = torch.softmax(torch.relu(torch.matmul(batch_adpvec, attns_sumT)), dim=-1)
            res = self.adpr * adp_a + (1 - self.adpr) * res
            res = torch.sum(torch.cat(torch.split(res.unsqueeze(-1), res.shape[0] // self.h, 0), -1), dim=-1)
        if dyna_adj != None:
            res += dyna_adj * formal_rate

        return res




class Gconv(nn.Module):
    def __init__(self, order=4):
        super().__init__()
        self.order = order

    def forward(self, dynadj, x):
        for i in range(self.order):
            x = torch.einsum("ncvl,nvw->ncwl", (x, dynadj)).contiguous()
        return x


class GAT_layer(nn.Module):
    def __init__(self, heads, in_dim, out_dim, samples, static_adj, dropout):
        super().__init__()
        features = in_dim
        self.h = heads
        self.d = features // heads
        self.s = samples
        self.statica = static_adj

        self.dyna_graph_layer = Dynamic_G_with_Attention(self.statica, int(self.s * math.log(self.statica.shape[0], 2)))

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])
        self.gconv = Gconv()
        self.ln = nn.LayerNorm(features, eps=1e-5)

        self.ff = FeedForward([features, features, out_dim], True)
        self.proj = nn.Linear(self.statica.shape[1], 1)

        self.dpt = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: [B,T,N,D]
        return: [B,T,N,D]
        '''
        Q = self.qfc(x)
        K = self.kfc(x)
        V = self.vfc(x)

        Q = torch.cat(torch.split(Q, self.d, -1), 0)
        K = torch.cat(torch.split(K, self.d, -1), 0)
        V = torch.cat(torch.split(V, self.d, -1), 0)

        B, T, N, D = K.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.statica, :]
        # K_sample = K_expand
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        Sampled_Nodes = int(self.s * math.log(N, 2))
        M = self.proj(Q_K_sample).squeeze(-1)
        # M_md = torch.median(M, dim=-1).values.unsqueeze(-1)
        # top_pos = torch.nonzero(M > M_md, as_tuple=True)[2].view(B, T, -1)
        # btm_pos = torch.nonzero(M < M_md, as_tuple=True)[2].view(B, T, -1)
        mediam = int((N - Sampled_Nodes) / 2)
        # _, sort_M = torch.sort(M, dim=-1)
        # top_pos = sort_M[:, :, mediam:-Sampled_Nodes]
        # btm_pos = sort_M[:, :, :mediam]
        sort_M_value, sort_M_idx = torch.sort(M, dim=-1)
        top_pos_v = torch.softmax(sort_M_value[:, :, mediam:-Sampled_Nodes].clamp(min=0), dim=-1)
        top_pos_i = sort_M_idx[:, :, mediam:-Sampled_Nodes]
        btm_pos_v = torch.softmax(sort_M_value[:, :, :mediam].clamp(min=0), dim=-1)
        btm_pos_i = sort_M_idx[:, :, :mediam]
        top_pos = torch.multinomial(
            top_pos_v.reshape(-1, top_pos_v.size(-1)),
            num_samples=int(Sampled_Nodes / 2),
            replacement=False
        )
        btm_pos = torch.multinomial(
            btm_pos_v.reshape(-1, btm_pos_v.size(-1)),
            num_samples=int(Sampled_Nodes / 2),
            replacement=False
        )
        M_rdm_t = torch.gather(top_pos_i.reshape(-1, top_pos_i.size(-1)), 1, top_pos).view(*top_pos_v.shape[:-1], -1)
        M_rdm_b = torch.gather(btm_pos_i.reshape(-1, btm_pos_i.size(-1)), 1, btm_pos).view(*btm_pos_v.shape[:-1], -1)

        # M_arg = list(range(int(btm_pos.shape[-1])))
        # random.shuffle(M_arg)
        # M_rdm = M_arg[:int(Sampled_Nodes / 2)]
        # M_rdm_t = top_pos[torch.arange(B)[:, None, None], torch.arange(T)[None, :, None], M_rdm]
        # M_rdm_b = btm_pos[torch.arange(B)[:, None, None], torch.arange(T)[None, :, None], M_rdm]
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]
        M_sample = torch.cat((M_top, M_rdm_t, M_rdm_b), dim=-1)
        # M_sample = M_top

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_sample, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        Q_K /= (self.d ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        dyna_adj = self.dyna_graph_layer(attn, cp)
        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_sample.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)
        dyna_adj = torch.sum(torch.cat(torch.split(dyna_adj.unsqueeze(-1), dyna_adj.shape[0] // self.h, 0), -1), dim=-1)

        value = self.ofc(value)
        # value = self.gconv(dyna_adj, value)

        value = self.dpt(value)
        value = self.ln(value)

        return self.ff(value), dyna_adj


class GAT_layer_v2(nn.Module):
    def __init__(self, heads, in_dim, out_dim, samples, static_adj, dropout):
        super().__init__()
        features = in_dim
        self.h = heads
        self.d = features // heads
        self.s = samples
        self.statica = static_adj

        self.dyna_graph_layer = Dynamic_G_with_Attention(self.statica, int(self.s * math.log(self.statica.shape[0], 2)), heads)

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])
        # self.gconv = Gconv()
        self.ln = nn.LayerNorm(features, eps=1e-5)

        self.ff = FeedForward([features, features, out_dim], True)
        self.proj = nn.Linear(self.statica.shape[1], 1)

        self.dpt = nn.Dropout(dropout)

    def forward(self, x, lda):
        '''
        x: [B,T,N,D]
        return: [B,T,N,D]
        '''
        Q = self.qfc(x)
        K = self.kfc(x)
        V = self.vfc(x)

        Q = torch.cat(torch.split(Q, self.d, -1), 0)
        K = torch.cat(torch.split(K, self.d, -1), 0)
        V = torch.cat(torch.split(V, self.d, -1), 0)
        dyna_adj = None

        B, T, N, D = K.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.statica, :]
        # K_sample = K_expand
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        Sampled_Nodes = int(self.s * math.log(N, 2))
        M = self.proj(Q_K_sample).squeeze(-1)
        # M_md = torch.median(M, dim=-1).values.unsqueeze(-1)
        # top_pos = torch.nonzero(M > M_md, as_tuple=True)[2].view(B, T, -1)
        # btm_pos = torch.nonzero(M < M_md, as_tuple=True)[2].view(B, T, -1)
        mediam = int((N - Sampled_Nodes) / 2)
        # _, sort_M = torch.sort(M, dim=-1)
        # top_pos = sort_M[:, :, mediam:-Sampled_Nodes]
        # btm_pos = sort_M[:, :, :mediam]
        sort_M_value, sort_M_idx = torch.sort(M, dim=-1)
        top_pos_v = torch.softmax(sort_M_value[:, :, mediam:-Sampled_Nodes].clamp(min=0), dim=-1)
        top_pos_i = sort_M_idx[:, :, mediam:-Sampled_Nodes]
        btm_pos_v = torch.softmax(sort_M_value[:, :, :mediam].clamp(min=0), dim=-1)
        btm_pos_i = sort_M_idx[:, :, :mediam]
        top_pos = torch.multinomial(
            top_pos_v.reshape(-1, top_pos_v.size(-1)),
            num_samples=int(Sampled_Nodes / 2),
            replacement=False
        )
        btm_pos = torch.multinomial(
            btm_pos_v.reshape(-1, btm_pos_v.size(-1)),
            num_samples=int(Sampled_Nodes / 2),
            replacement=False
        )
        M_rdm_t = torch.gather(top_pos_i.reshape(-1, top_pos_i.size(-1)), 1, top_pos).view(*top_pos_v.shape[:-1], -1)
        M_rdm_b = torch.gather(btm_pos_i.reshape(-1, btm_pos_i.size(-1)), 1, btm_pos).view(*btm_pos_v.shape[:-1], -1)

        # M_arg = list(range(int(btm_pos.shape[-1])))
        # random.shuffle(M_arg)
        # M_rdm = M_arg[:int(Sampled_Nodes / 2)]
        # M_rdm_t = top_pos[torch.arange(B)[:, None, None], torch.arange(T)[None, :, None], M_rdm]
        # M_rdm_b = btm_pos[torch.arange(B)[:, None, None], torch.arange(T)[None, :, None], M_rdm]
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]
        M_sample = torch.cat((M_top, M_rdm_t, M_rdm_b), dim=-1)
        # M_sample = M_top

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_sample, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        Q_K /= (self.d ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        # dyna_adj = self.dyna_graph_layer(attn, cp, dyna_adj=lda)
        # dyna_adj = torch.sum(torch.cat(torch.split(dyna_adj.unsqueeze(-1), dyna_adj.shape[0] // self.h, 0), -1), dim=-1)
        # value = self.gconv(dyna_adj, V)
        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_sample.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)

        value = self.ofc(value)
        # value = self.gconv(dyna_adj, value)

        value = self.dpt(value)
        value = self.ln(value)

        return self.ff(value), dyna_adj


class GAT_layer_v3(nn.Module):
    def __init__(self, heads, in_dim, out_dim, samples, static_adj, dropout):
        super().__init__()
        features = in_dim
        self.h = heads
        self.d = features // heads
        self.s = samples
        self.statica = static_adj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])
        self.ln = nn.LayerNorm(features, eps=1e-5)
        self.ff = FeedForward([features, features, out_dim], True)
        self.proj = nn.Linear(self.statica.shape[1], 1)
        self.dpt = nn.Dropout(dropout)

        self.sqfc = FeedForward([features, features])
        self.skfc = FeedForward([features, features])
        self.svfc = FeedForward([features, features])
        self.sofc = FeedForward([features, features])

    def forward(self, x):
        '''
        x: [B,T,N,D]
        return: [B,T,N,D]
        '''
        Q = self.qfc(x)
        Q_c = Q.clone()
        K = self.kfc(x)
        # K_c = K.clone()
        V = self.vfc(x)

        Q = torch.cat(torch.split(Q, self.d, -1), 0)
        K = torch.cat(torch.split(K, self.d, -1), 0)
        V = torch.cat(torch.split(V, self.d, -1), 0)
        dyna_adj = None

        B, T, N, D = K.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.statica, :]
        # K_sample = K_expand
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        Sampled_Nodes = int(self.s * math.log(N, 2))
        M = self.proj(Q_K_sample).squeeze(-1)
        mediam = int((N - Sampled_Nodes) / 2)
        sort_M_value, sort_M_idx = torch.sort(M, dim=-1)
        top_pos_v = torch.softmax(sort_M_value[:, :, mediam:-Sampled_Nodes].clamp(min=0), dim=-1)
        top_pos_i = sort_M_idx[:, :, mediam:-Sampled_Nodes]
        btm_pos_v = torch.softmax(sort_M_value[:, :, :mediam].clamp(min=0), dim=-1)
        btm_pos_i = sort_M_idx[:, :, :mediam]
        top_pos = torch.multinomial(
            top_pos_v.reshape(-1, top_pos_v.size(-1)),
            num_samples=int(Sampled_Nodes / 2),
            replacement=False
        )
        btm_pos = torch.multinomial(
            btm_pos_v.reshape(-1, btm_pos_v.size(-1)),
            num_samples=int(Sampled_Nodes / 2),
            replacement=False
        )
        M_rdm_t = torch.gather(top_pos_i.reshape(-1, top_pos_i.size(-1)), 1, top_pos).view(*top_pos_v.shape[:-1], -1)
        M_rdm_b = torch.gather(btm_pos_i.reshape(-1, btm_pos_i.size(-1)), 1, btm_pos).view(*btm_pos_v.shape[:-1], -1)

        M_top = M.topk(Sampled_Nodes, sorted=False)[1]
        M_sample = torch.cat((M_top, M_rdm_t, M_rdm_b), dim=-1)
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_sample, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        Q_K /= (self.d ** 0.5)
        attn = torch.softmax(Q_K, dim=-1)
        # cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        value = torch.matmul(attn, V)

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)
        projector = torch.cat(torch.split(Q_reduce, Q_reduce.shape[0] // self.h, 0), -1)
        M_pro = value.clone()
        K_reduce = K[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_sample, :]
        K_reduce = torch.cat(torch.split(K_reduce, K_reduce.shape[0] // self.h, 0), -1)
        A_pro = torch.softmax(torch.matmul(Q_c, K_reduce.transpose(-1, -2)) / (self.d ** 0.5), dim=-1)

        # sQ = self.sqfc(x)
        sK = self.skfc(projector)
        sV = self.svfc(value)
        # sQ = torch.cat(torch.split(sQ, self.d, -1), 0)
        sK = torch.cat(torch.split(sK, self.d, -1), 0)
        sV = torch.cat(torch.split(sV, self.d, -1), 0)

        sQK = torch.softmax(torch.matmul(Q, sK.transpose(-1, -2)) / (self.d ** 0.5), dim=-1)
        value = torch.matmul(sQK, sV)
        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)
        value = self.ofc(value)
        value = self.dpt(value)
        value = self.ln(value)
        value = self.ff(value)

        return value, A_pro, M_pro
