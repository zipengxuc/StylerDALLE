import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=F.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class MultiHeadAttentionQKV(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys = nn.Linear(dim_ref, dim_self, bias=bias)
        self.to_values = nn.Linear(dim_ref, dim_self, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, q, k, v, mask=False):
        b, n, c = q.shape
        _, m, d = k.shape
        # b n h dh
        queries = self.to_queries(q).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys = self.to_keys(k).reshape(b, m, self.num_heads, c // self.num_heads)
        values = self.to_values(v).reshape(b, m, self.num_heads, c // self.num_heads)
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask:
            mask = torch.eye(n, dtype=torch.bool).cuda()
            mask = mask.unsqueeze(0)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayerQKV(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_self)
        self.attn = MultiHeadAttentionQKV(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def __call__(self, input, mask=False, pos=0):
        (q, k, v) = input
        x = input[pos]
        x = x + self.attn(self.norm1(q), k, v, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLayer(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=F.relu):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def __call__(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=F.relu, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act))
        self.layers = nn.ModuleList(layers)

    def __call__(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerPos(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=F.relu):
        super(TransformerPos, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        layers = []
        for i in range(num_layers):
            # 1st-layer self
            layers.append(TransformerLayerQKV(dim_self, dim_self, num_heads, mlp_ratio, act=act))
            # 2nd-layer pos
            layers.append(TransformerLayerQKV(dim_self, dim_self, num_heads, mlp_ratio, act=act))
            # 3rd-layer cross
            layers.append(TransformerLayerQKV(dim_self, dim_ref, num_heads, mlp_ratio, act=act))
        self.layers = nn.ModuleList(layers)

    def __call__(self, dec, pos, enc):
        for i, layer in enumerate(self.layers):
            if i % 3 == 0:  # self
                dec = layer((dec, dec, dec), mask=True)
            elif (i+1) % 3 == 0:  # cross
                dec = layer((dec, enc, enc), mask=False)
            else:  # pos
                dec = layer((pos, pos, dec), mask=True, pos=2)

        return dec


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256, res_length=28):
        super().__init__()
        self.row_embed = nn.Embedding(res_length, num_pos_feats)
        self.col_embed = nn.Embedding(res_length, num_pos_feats)
        self.reset_parameters()
        self.num_pos_feats = num_pos_feats
        self.res_len = res_length

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, device):
        i = torch.arange(self.res_len, device=device)
        j = torch.arange(self.res_len, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(self.res_len, 1, 1),
            y_emb.unsqueeze(1).repeat(1, self.res_len, 1),
        ], dim=-1).view(-1, self.num_pos_feats*2).unsqueeze(0)
        return pos


class NATransformerPos(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, num_layers: int = 8,
                 res_length: int = 14, dim_ref: int = 512):
        super(NATransformerPos, self).__init__()
        self.res_length = res_length
        self.transformer = TransformerPos(dim_hidden, num_heads, num_layers, dim_ref=dim_ref)
        self.ln_final = nn.LayerNorm(dim_hidden)
        self.pos_embedding = PositionEmbeddingLearned(dim_hidden//2, res_length)

    def __call__(self, x, y):
        bs = x.size(0)
        pos = self.pos_embedding(x.device).repeat(bs, 1, 1)
        x = self.transformer(x, pos, y)  # (B, 256, 1024)
        x = self.ln_final(x)

        return x


class TransformerAddPos(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=F.relu, enc_dec: bool = False):
        super(TransformerAddPos, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        self.pos_embedding = PositionEmbeddingLearned(dim_self//2, 16)
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act))
        self.layers = nn.ModuleList(layers)

    def __call__(self, x, y=None, mask=None):
        bs = x.size(0)
        pos = self.pos_embedding(x.device).repeat(bs, 1, 1)
        x = x + pos
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, num_layers: int = 8, clip_length: int = 1):
        super(TransformerMapper, self).__init__()
        self.transformer = Transformer(dim_hidden, num_heads, num_layers)
        self.ln_final = nn.LayerNorm(dim_hidden)

    def __call__(self, x):
        x = self.transformer(x)  # (B, 256, 1024)
        x = self.ln_final(x)
        return x


class StylerDALLEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 8, res_length: int = 1):
        super(StylerDALLEModel, self).__init__()
        self.hidden_size = hidden_dim
        self.linear_proj = nn.Linear(input_dim, self.hidden_size)
        self.num_heads = num_heads
        self.res_len = res_length
        self.nat_enc = TransformerAddPos(self.hidden_size, num_heads, 4)
        self.nat_dec = NATransformerPos(hidden_dim, num_heads, num_layers, res_length, dim_ref=hidden_dim)
        self.outNet = nn.Linear(self.hidden_size, 8192)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def __call__(self, input_encodings):
        bs = input_encodings.size(0)
        scale = 2
        input_encodings = F.relu(self.linear_proj(input_encodings))
        y = self.nat_enc(input_encodings)
        input_encodings = input_encodings.view(bs, 16, 16, self.hidden_size)
        x_ = torch.repeat_interleave(input_encodings, scale, dim=1).view(bs, -1, self.hidden_size)  # bs, 49*4, 768
        x_ = torch.repeat_interleave(x_.view(bs, 16, 16*scale, self.hidden_size), scale, dim=1).view(bs, -1, self.hidden_size)
        outputs = F.dropout(self.nat_dec(x_, y)).view(bs, -1, self.hidden_size)
        output_size = outputs.size()
        flat_outputs = outputs.contiguous().view(-1, output_size[2])
        flat_scores = self.outNet(flat_outputs)
        flat_log_probs = self.logSoftmax(flat_scores)
        log_probs = flat_log_probs.contiguous().view(output_size[0], output_size[1], -1)
        return log_probs

