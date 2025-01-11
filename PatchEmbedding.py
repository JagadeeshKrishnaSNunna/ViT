import torch
from torch import nn
from einops import rearrange,repeat


class PatchEmbedding(nn.Module):
  def __init__(self,config):
    super().__init__()
    original_height=config['original_height']
    original_width=config['original_width']
    original_channels=config['channels']
    self.patch_height=config['patch_height']
    self.patch_width=config['patch_width']
    patch_embed_dimension=config['patch_embed_dimension']
    patch_embed_dropout=config['patch_embed_drop']

    self.number_patchs=(original_height//self.patch_height)*(original_width//self.patch_width)

    self.patch_embed=nn.Sequential(
     nn.Linear((original_channels*self.patch_height*self.patch_width),patch_embed_dimension)
     )
    
    self.pos_embed=nn.Parameter(torch.zeros(1,self.number_patchs+1,patch_embed_dimension))
    self.cls=torch.randn(1,patch_embed_dimension)
    self.dropout=nn.Dropout(patch_embed_dropout)
  
  def forward(self,x):
    batch_size=x.shape[0]
    patches=rearrange(x,'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',ph=self.patch_height,pw=self.patch_width)
    out=self.patch_embed(patches)
    out=torch.cat((repeat(self.cls.to(x.device),'x y -> b x y',b=batch_size),out),dim=1)
    out+=self.pos_embed
    out=self.dropout(out)
    return out

config={
    'original_height':224,
    'original_width':224,
    'channels':3,
    'patch_height':16,
    'patch_width':16,
    'patch_embed_dimension':4,
    'patch_embed_drop':0.0

}
model=PatchEmbedding(config)
test=torch.randn(1,3,224,224)
res=model(test)
assert model.number_patchs==196
