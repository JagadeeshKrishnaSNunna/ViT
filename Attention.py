import torch
from torch import nn
from einops import rearrange

class Attention(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.heads=config['heads']
    self.head_dimension=config['head_dimension']
    self.patch_embed_dimension=config['patch_embed_dimension']
    self.qkv_prj=nn.Linear(self.patch_embed_dimension,(self.heads*3*self.head_dimension))
    self.out_proj=nn.Linear(self.heads*self.head_dimension,self.patch_embed_dimension)

  def forward(self,x):
    batch,num_patches=x.shape[:2]
    q,k,v=self.qkv_prj(x).split(self.heads*self.head_dimension,dim=-1)
    
    q=rearrange(q,'b np (h hd)-> b h np hd',b=batch,np=num_patches,h=self.heads)
    k=rearrange(k,'b np (h hd)-> b h np hd',b=batch,np=num_patches,h=self.heads)
    v=rearrange(v,'b np (h hd)-> b h np hd',b=batch,np=num_patches,h=self.heads)
    
    att=nn.functional.softmax((torch.matmul(q,v.transpose(-2,-1))*(self.head_dimension**(-0.5))),dim=-1)
    att=torch.matmul(att,v)
    att=rearrange(att,'b h np hd -> b np (h hd)')
    return self.out_proj(att)


config={
    'heads':2,
    'head_dimension':2,
    'patch_embed_dimension':2
}
model=Attention(config)
test=torch.randn(1,2,2)
res=model(test)
assert test.shape == res.shape
