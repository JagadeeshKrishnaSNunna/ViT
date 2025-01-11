import torch
from torch import nn
from einops import rearrange,repeat
import matplotlib.pyplot as plt

import tqdm
from sklearn.metrics import confusion_matrix ,recall_score,precision_score,f1_score
import seaborn as sns

from ViT_Classifier import ViT_Classifier
from data import trainloader, testloader

device="cuda" if torch.cuda.is_available() else 'cpu'
config={
    'original_height':28,
    'original_width':28,
    'channels':1,
    'patch_height':4,
    'patch_width':4,
    'patch_embed_dimension':8,
    'patch_embed_drop':0.0,
    'heads':8,
    'head_dimension':8,
    'intermediate_nodes':32, #4*emb_dimension
    'num_transformerBlock':4,
    'num_classes':10,
    'fc_dropout':0.2,
    'lr':0.003,
    'epoch':1
}

model=ViT_Classifier(config).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), config['lr'])
losses=[]

def train_function(train_loader):
    model.train()
    loss_per_eppoc=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        loss_per_eppoc+=loss.item()
        
    return loss_per_eppoc/len(images)

for i in tqdm.tqdm(range(config['epoch'])):
  losses.append(train_function(trainloader))

plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.savefig('loss.png')

def validate(testloader):
  for images,labels in testloader:
    images=images.to(device)
    labels=labels
    with torch.no_grad():
     outputs = model(images)
    cm = confusion_matrix(labels, outputs.argmax(-1).cpu())
    recal=recall_score(labels, outputs.argmax(-1).cpu(),average='macro',zero_division=0)
    precision=precision_score(labels, outputs.argmax(-1).cpu(),average='macro',zero_division=0)
    f1=f1_score(labels, outputs.argmax(-1).cpu(),average='macro',zero_division=0)
  return cm,precision,recal,f1

confusionMatrix,precision,recal,f1=validate(testloader)

print("precision: ",precision)
print("recal: ",recal)
print("f1-score: ",f1)



sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')
