"""Revision model: batch-first SDPA, one residual, strict vanilla-base QKV conversion."""
from __future__ import annotations
import importlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from iss.models.sara import SaRALinear


class Attention(nn.Module):
    def __init__(self, dim, heads, shared, specific):
        super().__init__()
        self.heads = heads
        self.dim = dim
        for name in ('q_proj','k_proj','v_proj','out_proj'):
            setattr(self,name,SaRALinear(dim,dim,r=shared+specific,r_shared=shared))

    def forward(self, x):
        b,t,d = x.shape
        q,k,v = [getattr(self,name)(x).view(b,t,self.heads,d//self.heads).transpose(1,2)
                 for name in ('q_proj','k_proj','v_proj')]
        y = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
        return self.out_proj(y.transpose(1,2).contiguous().view(b,t,d))


def install_layer(shared, specific):
    class Layer(nn.Module):
        def __init__(self,d_ffn,nhead,d_model,dropout=0,activation=nn.ReLU,
                     normalize_before=False,attention_type='regularMHA',**kw):
            super().__init__()
            assert attention_type == 'regularMHA' and dropout == 0
            self.self_att = nn.Module()
            self.self_att.att = Attention(d_model,nhead,shared,specific)
            self.pos_ffn = nn.Module()
            self.pos_ffn.ffn = nn.Sequential(
                SaRALinear(d_model,d_ffn,r=shared+specific,r_shared=shared),
                activation(),nn.Dropout(dropout),
                SaRALinear(d_ffn,d_model,r=shared+specific,r_shared=shared))
            from speechbrain.nnet.normalization import LayerNorm
            self.norm1 = LayerNorm(d_model,eps=1e-6)
            self.norm2 = LayerNorm(d_model,eps=1e-6)
            self.normalize_before = normalize_before

        def body(self,src):
            x = self.norm1(src) if self.normalize_before else src
            src = src + self.self_att.att(x)
            if not self.normalize_before: src = self.norm1(src)
            x = self.norm2(src) if self.normalize_before else src
            src = src + self.pos_ffn.ffn(x)
            return src if self.normalize_before else self.norm2(src)

        def forward(self,src,src_mask=None,src_key_padding_mask=None,pos_embs=None):
            assert src_mask is None and src_key_padding_mask is None and pos_embs is None
            out = checkpoint(self.body,src,use_reentrant=False) if self.training and torch.is_grad_enabled() else self.body(src)
            return out,None
    module = importlib.import_module('speechbrain.lobes.models.transformer.Transformer')
    module.TransformerEncoderLayer = Layer


def build_model(num_spks,shared,specific,architecture=None):
    architecture = architecture or {}
    channels = int(architecture.get('channels',256))
    kernel_size = int(architecture.get('kernel_size',16))
    stride = int(architecture.get('stride',8))
    intra_layers = int(architecture.get('intra_layers',8))
    inter_layers = int(architecture.get('inter_layers',8))
    heads = int(architecture.get('heads',8))
    d_ffn = int(architecture.get('d_ffn',1024))
    dual_path_layers = int(architecture.get('dual_path_layers',2))
    chunk_size = int(architecture.get('chunk_size',250))
    install_layer(shared,specific)
    from iss.models.dual_path import Encoder,Decoder,SBTransformerBlock,Dual_Path_Model
    def block(num_layers):
        return SBTransformerBlock(num_layers=num_layers,d_model=channels,nhead=heads,d_ffn=d_ffn,
                                  dropout=0,use_positional_encoding=True,norm_before=True)
    return nn.ModuleDict({'encoder':Encoder(kernel_size=kernel_size,out_channels=channels),
        'masknet':Dual_Path_Model(in_channels=channels,out_channels=channels,
                                  intra_model=block(intra_layers),
                                  inter_model=block(inter_layers),num_layers=dual_path_layers,K=chunk_size,num_spks=num_spks,
                                  norm='ln',linear_layer_after_inter_intra=False,skip_around_intra=True),
        'decoder':Decoder(in_channels=channels,out_channels=1,kernel_size=kernel_size,stride=stride,bias=False)})


def convert_base(source, target):
    """Explicit in_proj split; reject every other unmatched source/destination key."""
    converted = {}
    mapping = []
    for key,value in source.items():
        if '.self_att.att.in_proj_' in key:
            prefix,kind = key.rsplit('in_proj_',1)
            assert kind in ('weight','bias')
            q_target = target[prefix+'q_proj.'+kind]
            assert value.shape[0] == q_target.shape[0] * 3
            for i,name in enumerate(('q_proj','k_proj','v_proj')):
                dest = prefix+name+'.'+kind
                converted[dest] = value.chunk(3,dim=0)[i].contiguous()
                mapping.append({'source':key,'target':dest,'split_index':i})
        else:
            converted[key] = value
    extra = set(converted)-set(target)
    missing = set(target)-set(converted)
    assert not extra, f'Unmapped base keys: {sorted(extra)[:10]}'
    assert all('sara_' in key for key in missing), f'Missing non-adapter keys: {sorted(missing)[:10]}'
    for key in missing: converted[key] = target[key]
    assert all(converted[k].shape == target[k].shape for k in target)
    return converted,mapping


def forward(model,mix):
    enc = model['encoder'](mix)
    masks = model['masknet'](enc)
    prediction = torch.stack([model['decoder'](enc*masks[i]) for i in range(masks.shape[0])],dim=-1)
    if prediction.shape[1] < mix.shape[1]:
        prediction = F.pad(prediction,(0,0,0,mix.shape[1]-prediction.shape[1]))
    return prediction[:,:mix.shape[1]]


def reset_specific(model,generator):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'sara_' not in name or '_spec' not in name: continue
            if 'sara_A_' in name: p.zero_()
            elif 'sara_B_' in name: nn.init.kaiming_uniform_(p,a=math.sqrt(5),generator=generator)
            else: raise ValueError(name)


def adapter_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if 'sara_' in k}


def load_adapters(model,state):
    current = model.state_dict()
    assert set(state) == {k for k in current if 'sara_' in k}
    current.update(state)
    model.load_state_dict(current,strict=True)
