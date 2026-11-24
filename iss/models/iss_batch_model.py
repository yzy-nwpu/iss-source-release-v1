"""Same model mathematics with explicitly configurable activation checkpoints."""
from types import MethodType
import torch
from torch.utils.checkpoint import checkpoint
from iss.models.iss_model import build_model as original_build_model
from iss.models.iss_model import convert_base,forward,reset_specific,adapter_state,load_adapters


def memory_forward(self,src,src_mask=None,src_key_padding_mask=None,pos_embs=None):
    assert src_mask is None and src_key_padding_mask is None and pos_embs is None
    if self.training and torch.is_grad_enabled() and self._activation_checkpoint:
        out=checkpoint(self.body,src,use_reentrant=False)
    else:
        out=self.body(src)
    return out,None


def build_model(num_spks,shared,specific,checkpoint_layers=32,architecture=None):
    model=original_build_model(num_spks,shared,specific,architecture)
    layers=[m for m in model.modules() if type(m).__name__=='Layer']
    assert layers and 0<=checkpoint_layers<=len(layers)
    for i,layer in enumerate(layers):
        layer._activation_checkpoint=i<checkpoint_layers
        layer.forward=MethodType(memory_forward,layer)
    return model
