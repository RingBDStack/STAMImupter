import torch
from torch import nn
from torch.nn import functional as F
from model.GAT import GAT_layer, GAT_layer_v2, GAT_layer_v3
# from Temporal_Transformer import Encoder_only_Temptrans
from torch.nn import TransformerEncoderLayer
# from tsl.nn.blocks.encoders import TransformerLayer
# from tsl.nn.blocks.encoders.mlp import MLP
import math


def generate_sinusoidal_encoding(dim, length):
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding  # Shape: (length, dim)


def get_mask_embed(x):
    b, t, n = x.shape
    mt = (x != 0).sum(dim=-1) / x.size(-1)
    ms = (x.transpose(-1, -2) != 0).sum(dim=-1) / x.size(-2)
    mt = mt.unsqueeze(-1).expand(b, t, n).unsqueeze(-1)
    ms = ms.unsqueeze(-2).expand(b, t, n).unsqueeze(-1)

    return mt, ms


class Dynamic_G_with_Attention_v2(nn.Module):
    def __init__(self, embed_dim, num_nodes, hidden_dim):
        super().__init__()
        self.ed = embed_dim * 2
        self.adpvec = nn.Parameter(torch.randn(embed_dim, num_nodes), requires_grad=True)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def toph(self, x, r=0.5):
        x_sort, _ = torch.sort(x, dim=-1, descending=True)
        length = int(x.shape[-1] * r)
        th = x_sort[..., length - 1]
        res = torch.where(x >= th.unsqueeze(-1), x, torch.zeros_like(x))
        return res

    def forward(self, A, M):
        M = self.MLP(M[:, :, :self.ed])
        E_ref = self.adpvec.unsqueeze(0).expand(M.shape[0], *self.adpvec.shape)
        E_adp = self.toph(torch.matmul(M, E_ref))
        A_adp = torch.softmax(torch.relu(torch.matmul(A.transpose(-1, -2), E_adp)), dim=-1)
        return A_adp

class SP_TSFormer_MoE(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_nodes, num_series,
                 adj, num_heads, mlp_ratio, dropout, num_layers, sample=1,
                 mask_mode="point", mask_rate=0.2, mode="train"):
        super().__init__()
        assert mask_mode in ["point", "block"], "Error mask mode."
        assert mode in ["train", "apply"], "Error mode."
        self.id = input_dim
        self.ed = embed_dim
        self.od = output_dim
        self.nh = num_heads
        self.nl = num_layers
        self.mask_mode = mask_mode
        self.mr = mask_rate
        self.mode = mode
        self.nn = num_nodes
        self.ns = num_series
        self.adj = adj
        self.device = "cuda:0"

        self.start_Conv_T = nn.Conv2d(input_dim, embed_dim, kernel_size=(1, 1))
        self.start_Conv_S = nn.Conv2d(input_dim, embed_dim, kernel_size=(1, 1))

        self.positional_encoding = nn.Parameter(torch.empty(self.ns, self.nn, embed_dim), requires_grad=True)

        self.MoE_layers = nn.ModuleList()
        self.Tlayers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, embed_dim*mlp_ratio, dropout)
                                      for i in range(num_layers)])
        self.Slayers = nn.ModuleList([GAT_layer(num_heads, embed_dim * 2, embed_dim, sample, adj, dropout)
                                      for i in range(num_layers)])
        # self.Slayers = nn.ModuleList([GAT_layer(num_heads, embed_dim, embed_dim, sample, adj, dropout)
        #                               for i in range(num_layers)])

        self.Gating_layer = nn.Linear(embed_dim, 2)
        self.outputlayer_T = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        nn.init.uniform_(self.positional_encoding, -.02, .02)

    def forward(self, x, xl, xh):
        x = torch.from_numpy(x).float().to(self.device)
        xl = torch.from_numpy(xl).float().to(self.device)
        xh = torch.from_numpy(xh).float().to(self.device)


        PE = self.positional_encoding.unsqueeze(0)
        Tin = self.start_Conv_T(torch.cat((x, xl, xh), dim=-1).transpose(1, 3)).transpose(1, 3)
        Sin = self.start_Conv_S(torch.cat((x, xl, xh), dim=-1).transpose(1, 3)).transpose(1, 3)
        moe_weights = F.softmax(self.Gating_layer(Tin), dim=-1)
        Tin = Tin + PE
        Sin = Sin + PE
        B, T, N, D = Sin.shape
        Touti, Souti = None, None

        for i in range(self.nl):
            Touti = self.Tlayers[i](Tin.transpose(1, 2).contiguous().view(B*N, T, D).transpose(0, 1))
            Touti = Touti.transpose(0, 1).contiguous().view(B, N, T, D).transpose(1, 2)
            Sin = torch.cat((Sin, Touti), dim=-1)
            # Sin = Touti
            Souti, dynadj = self.Slayers[i](Sin)
            Sin = Souti
            Tin = Touti

        # Touti = self.outputlayer_T(Touti)
        # Souti = self.outputlayer_S(Souti)
        outputs = torch.stack((Touti, Souti), dim=-2)
        outputs = (moe_weights.unsqueeze(-1) * outputs).sum(dim=-2)
        # outputs = Souti
        outputs = self.outputlayer_T(outputs)
        # outputs = Touti

        return outputs


class Gconv(nn.Module):
    def __init__(self, in_dim, order=2, drop=0.1):
        super().__init__()
        self.order = order
        self.l = nn.Linear(in_dim, in_dim)
        self.d = nn.Dropout(drop)
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, dynadj, x):
        res = x.clone()
        for i in range(self.order):
            x = torch.einsum("ncvl,nvw->ncwl", (x, dynadj)).contiguous()
        x = res + self.d(self.l(x))
        return self.ln(x)


class Gconv_static(nn.Module):
    def __init__(self, in_dim, order=2, drop=0.2):
        super().__init__()
        self.order = order
        self.l = nn.Linear(in_dim, in_dim)
        self.d = nn.Dropout(drop)

    def forward(self, dynadj, x):
        for i in range(self.order):
            x = torch.einsum("ncvl,vw->ncwl", (x, dynadj)).contiguous()
        x = self.d(self.l(x))
        return x



class SP_TSFormer_MoE_v2(nn.Module):
    def __init__(self, input_dim, embed_dim, Tembed_dim, output_dim, num_nodes, num_series,
                 adj, num_heads, mlp_ratio, dropout, num_layers, sample=1,
                 mask_mode="point", mask_rate=0.2, mode="train"):
        super().__init__()
        assert mask_mode in ["point", "block"], "Error mask mode."
        assert mode in ["train", "apply"], "Error mode."
        self.id = input_dim
        self.ed = embed_dim
        self.od = output_dim
        self.nh = num_heads
        self.nl = num_layers
        self.mask_mode = mask_mode
        self.mr = mask_rate
        self.mode = mode
        self.nn = num_nodes
        self.ns = num_series
        self.adj = adj
        self.device = "cuda:0"

        self.start_Conv_T = nn.Conv2d(input_dim, embed_dim, kernel_size=(1, 1))
        self.start_Conv_S = nn.Conv2d(input_dim, embed_dim, kernel_size=(1, 1))

        self.positional_encoding = nn.Parameter(torch.empty(self.ns, self.nn, embed_dim), requires_grad=True)

        self.Tlayers = nn.ModuleList([TransformerEncoderLayer(embed_dim * 2, num_heads, embed_dim * mlp_ratio, dropout)
                                      for i in range(num_layers)])
        self.Slayers = nn.ModuleList([GAT_layer_v3(num_heads, embed_dim * 4, embed_dim * 2, sample, adj, dropout)
                                      for i in range(num_layers)])
        # self.Slayers = nn.ModuleList([GAT_layer(num_heads, embed_dim, embed_dim, sample, adj, dropout)
        #                               for i in range(num_layers)])

        self.Gating_layer = nn.Linear(3, 2)
        self.dyna_layer = Dynamic_G_with_Attention_v2(embed_dim * 2, num_nodes, embed_dim * 4)
        self.outputlayer_T = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        nn.init.uniform_(self.positional_encoding, -.02, .02)
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True).to(self.device)
        # self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True).to(self.device)




    def forward(self, x, xl, xh):
        x = torch.from_numpy(x).float().to(self.device)
        xl = torch.from_numpy(xl).float().to(self.device)
        xh = torch.from_numpy(xh).float().to(self.device)


        PE = self.positional_encoding.unsqueeze(0).expand(x.shape[0], *self.positional_encoding.shape)
        mt, ms = get_mask_embed(x[:, :, :, 0])
        # Tin = self.start_Conv_T(torch.cat((x, xl, xh, mt, ms), dim=-1).transpose(1, 3)).transpose(1, 3)
        # Sin = self.start_Conv_S(torch.cat((x, xl, xh, mt, ms), dim=-1).transpose(1, 3)).transpose(1, 3)
        Tin = self.start_Conv_T(torch.cat((x, xl, xh), dim=-1).transpose(1, 3)).transpose(1, 3)
        Sin = self.start_Conv_S(torch.cat((x, xl, xh), dim=-1).transpose(1, 3)).transpose(1, 3)
        moe_weights = F.softmax(self.Gating_layer(torch.cat((x[:, :, :, :1], mt, ms), dim=-1)), dim=-1)
        # moe_weights = F.softmax(self.Gating_layer(Tin), dim=-1)
        Tin = torch.cat((Tin, PE), dim=-1)
        Sin = torch.cat((Sin, PE), dim=-1)
        B, T, N, D = Sin.shape
        Touti, Souti = None, None
        dynadj, A, M = None, None, None


        for i in range(self.nl):
            Touti = self.Tlayers[i](Tin.transpose(1, 2).contiguous().view(B*N, T, D).transpose(0, 1))
            Touti = Touti.transpose(0, 1).contiguous().view(B, N, T, D).transpose(1, 2)
            Sin = torch.cat((Sin, Touti), dim=-1)
            Souti, A, M = self.Slayers[i](Sin)
            # Touti = Touti + self.dyna_gconv(dynadj, Touti)
            Sin = Souti
            Tin = Touti

        # dynadj = self.dyna_layer(A[:, -1, :, :], M[:, -1, :, :])
        outputs = torch.stack((Touti, Souti), dim=-2)
        outputs = (moe_weights.unsqueeze(-1) * outputs).sum(dim=-2)
        # outputs = (moe_weights.unsqueeze(-1) * outputs).view(B, T, N, -1)
        # outputs = torch.cat((Touti, Souti), dim=-1)
        outputs = self.outputlayer_T(outputs)

        return outputs, A[:, -1, :, :], M[:, -1, :, :]

