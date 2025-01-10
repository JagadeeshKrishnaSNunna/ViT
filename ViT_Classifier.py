import torch
from torch import nn
from einops import rearrange,repeat
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import tqdm

class ViT_Classifier(nn.Module):
  def __init__(self,config):
    super().__init__()
    num_transformerBlock=config['num_transformerBlock']
    embed_dimension=config['patch_embed_dimension']
    num_classes=config['num_classes']

    self.Patch_embed=PatchEmbedding(config)
    self.Transformer_Blocks=nn.ModuleList(
        [TransformerBlock(config) for i in range(num_transformerBlock)]
        )

    self.classifier_out=nn.Linear(embed_dimension,num_classes)

  def forward(self,x):
    x=self.Patch_embed(x)
    for block in self.Transformer_Blocks:
      x=block(x)
    return self.classifier_out(x[:,0])
