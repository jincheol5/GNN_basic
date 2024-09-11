import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from gcn import GCN
from train_module import Trainer


### load dataset ###
## Cora dataset
# for unsupervised learning
# for node classification
# undirected graph
# num_node = 2708
# num_edge = 10556/2 = 5728
# num_classes = 7
# num_node_features = 1433
# train node = 140 
# validation node = 500
# test node = 1000

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Cora dataset load
dataset=Planetoid(root='/tmp/Cora',name='Cora')
data=dataset[0]
data.to(device) 

### model training ###
# load model
model=GCN(dataset.num_features,16,dataset.num_classes).to(device) # hidden layer feature = 16

# train
model_trainer=Trainer(model)
model_trainer.training(data,100) # epochs=100


