import torch
from torch import nn
from einops import rearrange,repeat


class FCLayers(nn.Module):
  def __init__(self,config):
    super().__init__()
    embed_dimension=config['patch_embed_dimension']
    intermediate_nodes=config['intermediate_nodes']
    dropout=config['fc_dropout']
    self.network=nn.Sequential(
        nn.Linear(embed_dimension,intermediate_nodes),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(intermediate_nodes,embed_dimension),
        nn.Dropout(dropout)
    )
  def forward(self,x):
    return self.network(x)

class TransformerBlock(nn.Module):
  def __init__(self,config):
    super().__init__()
    embed_dimension=config['patch_embed_dimension']
    intermediate_nodes=config['intermediate_nodes']
    self.att_block=nn.Sequential(
        nn.LayerNorm(embed_dimension),
        Attention(config)
        )
    self.FC_Block=nn.Sequential(
        nn.LayerNorm(embed_dimension),
        FCLayers(config)
    )

  def forward(self,x):
    x=x+self.att_block(x)
    x=x+self.FC_Block(x)
    return x
